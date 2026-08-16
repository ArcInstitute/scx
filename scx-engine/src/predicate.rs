// Predicate AST, parser, and evaluator for the SCX query engine.
//
// Grammar (recursive-descent):
//   expr       := or_expr
//   or_expr    := and_expr ("or" and_expr)*
//   and_expr   := not_expr ("and" not_expr)*
//   not_expr   := "not" atom | atom
//   atom       := comparison | in_expr | "(" expr ")"
//   comparison := IDENT ("==" | "!=" | "<" | ">" | "<=" | ">=") value
//   in_expr    := IDENT "in" "[" value ("," value)* "]"
//   value      := STRING | INTEGER | FLOAT | BOOL

use std::fmt;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, RecordBatch};
use arrow::compute;
use arrow::datatypes::{DataType, Schema};

use crate::error::{EngineError, Result};

// ---------------------------------------------------------------------------
// AST types (A2)
// ---------------------------------------------------------------------------

/// A scalar value used in predicate comparisons.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Utf8(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(s) => write!(f, "'{s}'"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

/// A predicate expression tree for filtering rows.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Eq(String, ScalarValue),
    Ne(String, ScalarValue),
    Lt(String, ScalarValue),
    Gt(String, ScalarValue),
    Le(String, ScalarValue),
    Ge(String, ScalarValue),
    In(String, Vec<ScalarValue>),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

impl Predicate {
    /// Returns all column names referenced by this predicate (including nested).
    pub fn columns(&self) -> Vec<&str> {
        match self {
            Self::Eq(col, _)
            | Self::Ne(col, _)
            | Self::Lt(col, _)
            | Self::Gt(col, _)
            | Self::Le(col, _)
            | Self::Ge(col, _)
            | Self::In(col, _) => vec![col.as_str()],
            Self::And(a, b) | Self::Or(a, b) => {
                let mut cols = a.columns();
                cols.extend(b.columns());
                cols
            }
            Self::Not(inner) => inner.columns(),
        }
    }
}

// ---------------------------------------------------------------------------
// Row-set evaluator (row-set predicate pushdown over the obs index)
//
// A second evaluator that walks the SAME `Predicate` AST as `eval_inner`, but
// instead of producing a boolean mask over a decoded `RecordBatch` it resolves
// the predicate to a global obs-row [`RowSet`] directly from the
// `PredicateIndex` — no obs-shard decode. Returns `None` for any subtree that
// cannot be resolved *exactly* from the index, so the caller routes that part
// through the existing decode-and-mask (residual) path.
//
// v1 exactness scope: categorical `Eq` / `In` and `And` / `Or` of those. `Ne`,
// `Not`, and all numeric comparisons return `None` (residual) — see the module
// docs in `collect.rs`: the categorical index omits null rows, so complementing
// it (for `Ne`/`Not`) would wrongly re-include them, and numeric B+ tree leaves
// are conservative.
//
// **Why the set algebra is exact.** Each leaf resolves to the rows where that
// leaf is TRUE (the index skips nulls). Kleene `AND` is TRUE iff both operands
// are TRUE and Kleene `OR` is TRUE iff either is, so `RowSet::intersect` /
// `RowSet::union` compute precisely the Kleene TRUE-set — which, after
// `evaluate`'s top-level UNKNOWN→false coalesce, is precisely the mask. This
// holds for nullable columns too, and it holds *only* because no subtree is
// ever evaluated under a negation: `Ne` and `Not` are residual here, and
// `partition_obs_predicates` splits only top-level `And` conjuncts. Keep that
// precondition — under a negation the TRUE-set is no longer sufficient (you
// would need the FALSE-set, which is not the complement when nulls exist).
// ---------------------------------------------------------------------------

use crate::index::PredicateIndex;
use crate::pushdown::CategoryDictionaries;
use crate::rowset::{shard_range_to_global, RowSet};

/// Context for [`eval_rowset`]: the obs predicate index plus the **shard
/// row-range table the index was built against** (`(shard_id, row_start,
/// row_end)`, sorted by `shard_id`) used to map shard-local ranges to global
/// rows. The predicate index is keyed to the CSR/output shard ranges
/// (`output_shard_row_ranges` in merge/compact/append, `csr_row_ranges` in
/// conversion) — NOT the obs-metadata-shard ranges, which use an independent
/// `shard_target_rows` chunking — so this table is the catalog's CSR shard
/// ranges.
pub struct RowSetCtx<'a> {
    pub index: &'a PredicateIndex,
    pub shard_row_ranges: &'a [(u32, u64, u64)],
    pub n_obs: u64,
    /// The same per-column vocabularies Level-1 prunes with, carried here for
    /// their `complete` bit alone.
    ///
    /// Level-2 reads a categorical miss the same way Level-1 does, and more
    /// sharply: [`PredicateIndex::categorical_eq`] returns `Some(&[])` for an
    /// absent value, which `eval_rowset` treats as an *exact* empty row-set
    /// rather than "not resolvable here". On a vocabulary that is not the
    /// column's complete value set that is a wrong answer, not a slow one, so
    /// an incomplete column is residual here — see [`RowSetCtx::resolvable`].
    pub category_dicts: &'a CategoryDictionaries,
}

impl RowSetCtx<'_> {
    /// Whether a categorical predicate on `column` may be resolved from the
    /// index. `false` only when the column IS indexed as a categorical and its
    /// vocabulary is not trustworthy as complete; an unindexed column is
    /// declined by `categorical_eq` itself.
    fn resolvable(&self, column: &str) -> bool {
        self.category_dicts.vocabulary_is_usable(column)
    }

    /// Map a slice of shard-local ranges to a global [`RowSet`]. Returns `None`
    /// if any range references a shard absent from the range table (stale
    /// index) — the caller then treats the predicate as residual.
    fn shard_ranges_to_rowset(&self, ranges: &[crate::index::ShardRange]) -> Option<RowSet> {
        let mut out = Vec::with_capacity(ranges.len());
        for sr in ranges {
            out.push(shard_range_to_global(sr, self.shard_row_ranges)?);
        }
        Some(RowSet::from_ranges(out))
    }
}

// ---------------------------------------------------------------------------
// The shared tree walk
//
// Both evaluators recurse over the same `Predicate` tree and differ only in
// what a node evaluates *to*: a Kleene `BooleanArray` over a decoded batch
// (`MaskAlgebra`) or an exactly-resolved `Option<RowSet>` read out of the index
// (`RowSetAlgebra`). Only the recursion is shared — each instance keeps its own
// leaves AND its own combinator kernels, because that is where they genuinely
// differ.
//
// ⚠️ **What this does and does not buy.** Be precise here, because two
// plausible-sounding claims are both false.
//
// It does NOT make the two evaluators agree, and it would NOT have prevented
// §5.1 (the row-set path deliberately restricted to reproduce the mask path's
// non-Kleene `or`). That was a *combinator kernel choice* inside one instance,
// and each instance still picks its own — as it must, since `or_kleene` and
// `RowSet::union` are not interchangeable. The thing that catches a divergence
// is the independent oracle in `tests/rowset_differential.rs`.
//
// It also does NOT add the "a new `Predicate` variant cannot be silently
// mishandled" property. That was already true: `eval_inner` and `eval_rowset`
// were each exhaustive with no `_` arm. Measured by adding a variant — five
// compile errors before this refactor, six after, the sixth being `walk`
// itself.
//
// What it buys is narrower: the descent is written once instead of twice, and
// the two evaluators' actual difference is now legible as what it is — seven
// leaves and three combinator kernels — rather than something you infer by
// diffing two fifty-line matches.
//
// The unification is deliberately not the `(definitely, maybe)` lattice that
// would make numeric leaves narrowable. That is a redesign, not a refactor: it
// needs B+ tree range narrowing the index does not have.
// ---------------------------------------------------------------------------

/// One interpretation of a [`Predicate`] tree. See the section header above.
trait PredicateAlgebra {
    type Value;

    /// Evaluate a non-combinator node. [`walk`] never passes `And` / `Or` /
    /// `Not` here.
    fn leaf(&self, pred: &Predicate) -> Result<Self::Value>;

    fn and(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value>;
    fn or(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value>;
    fn not(&self, a: Self::Value) -> Result<Self::Value>;

    /// Whether `v` already determines the enclosing `And` / `Or`, so the other
    /// operand need not be evaluated.
    ///
    /// This exists to preserve `eval_rowset`'s original `?` short-circuit
    /// exactly: a residual operand makes the whole node residual, and walking
    /// the sibling would be pure but pointless index work. `MaskAlgebra` takes
    /// the default — the mask evaluator never short-circuited, because both
    /// Kleene kernels need both operands.
    fn absorbing(&self, _v: &Self::Value) -> bool {
        false
    }

    /// A value for `pred` that needs no descent into it at all. `Some(v)` makes
    /// [`walk`] return `v` **without visiting `pred`'s operands**.
    ///
    /// `absorbing` cannot express this: it is consulted on a value that has
    /// already been computed, so it can skip a *sibling* but never the subtree
    /// under a unary node. `RowSetAlgebra` needs exactly that for `Not`, which
    /// is unconditionally residual — the categorical index omits NULL rows, so
    /// complementing its TRUE-set would wrongly re-include them, and no operand
    /// value can change that answer.
    ///
    /// Without this hook `not (cell_type == 'T cell')` resolves the inner leaf,
    /// turns every matching `ShardRange` into a `RowSet` (allocating, sorting
    /// and coalescing the ranges), throws it away, and *then* takes the residual
    /// decode path it was always going to take — O(matching shard ranges) of CPU
    /// and peak memory bought for nothing, on an atlas-scale indexed
    /// categorical. `partition_obs_predicates` flattens only `And`, so a
    /// negated conjunct reaches `eval_rowset` whole and this is reachable from a
    /// plain `filter_obs("not (cell_type == 'T cell')")`.
    fn short_circuit(&self, _pred: &Predicate) -> Option<Self::Value> {
        None
    }
}

/// Recurse over `pred` under `alg`.
///
/// The match is exhaustive over `Predicate` with no `_` arm, so a new variant
/// is a compile error here rather than being silently routed to `leaf` — where
/// a combinator would hit `unreachable!` at runtime.
fn walk<A: PredicateAlgebra>(pred: &Predicate, alg: &A) -> Result<A::Value> {
    if let Some(v) = alg.short_circuit(pred) {
        return Ok(v);
    }
    // Asked BEFORE any descent: an algebra that already knows this node's answer
    // must not pay for its operands. See `PredicateAlgebra::short_circuit`.
    match pred {
        Predicate::And(a, b) => {
            let left = walk(a, alg)?;
            if alg.absorbing(&left) {
                return Ok(left);
            }
            let right = walk(b, alg)?;
            alg.and(left, right)
        }
        Predicate::Or(a, b) => {
            let left = walk(a, alg)?;
            if alg.absorbing(&left) {
                return Ok(left);
            }
            let right = walk(b, alg)?;
            alg.or(left, right)
        }
        Predicate::Not(inner) => {
            let v = walk(inner, alg)?;
            alg.not(v)
        }
        Predicate::Eq(_, _)
        | Predicate::Ne(_, _)
        | Predicate::Lt(_, _)
        | Predicate::Gt(_, _)
        | Predicate::Le(_, _)
        | Predicate::Ge(_, _)
        | Predicate::In(_, _) => alg.leaf(pred),
    }
}

/// Resolve leaves straight from the obs predicate index. `None` is *residual* —
/// "this node is not exactly resolvable here" — not "no rows".
struct RowSetAlgebra<'a, 'b> {
    ctx: &'a RowSetCtx<'b>,
}

impl PredicateAlgebra for RowSetAlgebra<'_, '_> {
    type Value = Option<RowSet>;

    fn leaf(&self, pred: &Predicate) -> Result<Self::Value> {
        let ctx = self.ctx;
        Ok(match pred {
            Predicate::Eq(col, ScalarValue::Utf8(v)) => {
                // `categorical_eq` returns None iff the column is not an indexed
                // categorical (residual); Some(&[]) iff indexed but value absent
                // (exact empty row-set). That second reading is only a row-set at
                // all when the vocabulary is the column's complete value set.
                if !ctx.resolvable(col) {
                    return Ok(None);
                }
                let Some(ranges) = ctx.index.categorical_eq(col, v) else {
                    return Ok(None);
                };
                ctx.shard_ranges_to_rowset(ranges)
            }
            // Eq on a categorical column with a non-string literal cannot match
            // a string category; only string equality is index-resolvable here.
            Predicate::Eq(_, _) => None,
            Predicate::In(col, vals) => {
                // Only index-resolvable if the column is an indexed categorical
                // whose vocabulary is complete (see the `Eq` arm).
                if ctx.index.indexed_kind(col) != Some(crate::index::IndexKind::Categorical)
                    || !ctx.resolvable(col)
                {
                    return Ok(None);
                }
                // Every member must be resolvable, or the whole predicate is
                // residual. A non-string member against a string-valued
                // categorical index is *unknown*, not provably absent: skipping
                // it and returning the union of the members that did resolve
                // silently narrows the result. That is not hypothetical — a file
                // written by an earlier version indexed an integer-valued
                // categorical as an entry-less `CategoricalIndex`, so
                // `batch in [1, 2]` resolved to an exact **empty** row set.
                if vals.iter().any(|v| !matches!(v, ScalarValue::Utf8(_))) {
                    return Ok(None);
                }
                let mut acc = RowSet::empty();
                for v in vals {
                    if let ScalarValue::Utf8(s) = v {
                        let Some(ranges) = ctx.index.categorical_eq(col, s) else {
                            return Ok(None);
                        };
                        let Some(rs) = ctx.shard_ranges_to_rowset(ranges) else {
                            return Ok(None);
                        };
                        acc = acc.union(&rs);
                    }
                }
                Some(acc)
            }
            // Residual in v1 (see module docs): Ne (null-complement hazard),
            // numeric comparisons (conservative B+ tree leaves).
            //
            // A numeric leaf bounds a row range by `[min_value, max_value]`; it
            // does not say which row holds which value. So a range lookup yields
            // the rows that *may* match, and `eval_rowset` must return the rows
            // that *do* — `Some` here means exactly resolvable, and the caller
            // stops evaluating the predicate. Making these resolvable is not a
            // matter of indexing more finely: it needs the (definitely, maybe)
            // lattice that would let a residual-but-narrowed answer be expressed
            // at all. Until then, numeric predicates are narrowed at Level 1 by
            // the per-shard `MinMax` column stats and then evaluated on the mask
            // path, and the leaves are kept at shard granularity because that is
            // the only granularity anything reads them at.
            Predicate::Ne(_, _)
            | Predicate::Lt(_, _)
            | Predicate::Gt(_, _)
            | Predicate::Le(_, _)
            | Predicate::Ge(_, _) => None,
            // `walk` handles these and never routes them here.
            Predicate::And(_, _) | Predicate::Or(_, _) | Predicate::Not(_) => {
                unreachable!("walk dispatches combinators")
            }
        })
    }

    fn and(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value> {
        Ok(match (a, b) {
            (Some(ra), Some(rb)) => Some(ra.intersect(&rb)),
            _ => None,
        })
    }

    fn or(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value> {
        // An Or with a residual side can match rows in ANY shard, so it is not
        // narrowable: both sides must resolve exactly. That is the only
        // restriction — nullable operand columns are fine, because the union is
        // exactly the Kleene TRUE-set (see the module header).
        Ok(match (a, b) {
            (Some(ra), Some(rb)) => Some(ra.union(&rb)),
            _ => None,
        })
    }

    /// `Not` is residual in v1: the categorical index omits null rows, so
    /// complementing its TRUE-set would wrongly re-include them.
    ///
    /// Unreachable in practice — [`Self::short_circuit`] answers `Not` before
    /// `walk` descends, so this is never called. Kept correct rather than
    /// `unreachable!` so the algebra stays well-defined on its own terms.
    fn not(&self, _a: Self::Value) -> Result<Self::Value> {
        Ok(None)
    }

    fn absorbing(&self, v: &Self::Value) -> bool {
        v.is_none()
    }

    /// `Not` is residual whatever its operand resolves to, so the operand is
    /// never worth resolving. This restores the pre-refactor behaviour: the
    /// original `eval_rowset` matched `Predicate::Not(_)` in its residual arm
    /// and returned `None` without recursing.
    fn short_circuit(&self, pred: &Predicate) -> Option<Self::Value> {
        match pred {
            Predicate::Not(_) => Some(None),
            _ => None,
        }
    }
}

/// Resolve `pred` to an exact global [`RowSet`] using only the index, or `None`
/// if any node touches a non-indexed column or an op outside the v1 exact
/// scope (then the whole subtree is residual).
pub fn eval_rowset(pred: &Predicate, ctx: &RowSetCtx) -> Option<RowSet> {
    // `RowSetAlgebra` never returns `Err` — every leaf resolves to `Some`/`None`
    // from in-memory index lookups. If that ever changes, `None` is the safe
    // reading: it routes the predicate to the residual decode-and-mask path,
    // which is slower and always correct.
    walk(pred, &RowSetAlgebra { ctx }).unwrap_or(None)
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eq(col, val) => write!(f, "{col} == {val}"),
            Self::Ne(col, val) => write!(f, "{col} != {val}"),
            Self::Lt(col, val) => write!(f, "{col} < {val}"),
            Self::Gt(col, val) => write!(f, "{col} > {val}"),
            Self::Le(col, val) => write!(f, "{col} <= {val}"),
            Self::Ge(col, val) => write!(f, "{col} >= {val}"),
            Self::In(col, vals) => {
                let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
                write!(f, "{col} in [{}]", vals_str.join(", "))
            }
            Self::And(a, b) => write!(f, "({a} and {b})"),
            Self::Or(a, b) => write!(f, "({a} or {b})"),
            Self::Not(inner) => write!(f, "not {inner}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer (A3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum OpKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f64),
    BoolLit(bool),
    Op(OpKind),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    And,
    Or,
    Not,
    In,
}

fn tokenize(expr: &str) -> Result<Vec<Token>> {
    let chars: Vec<char> = expr.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        // String literal (single or double quotes)
        if chars[i] == '\'' || chars[i] == '"' {
            let quote = chars[i];
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != quote {
                i += 1;
            }
            if i >= chars.len() {
                return Err(EngineError::PredicateParseError {
                    expr: expr.to_string(),
                    reason: "unterminated string literal".to_string(),
                });
            }
            let s: String = chars[start..i].iter().collect();
            tokens.push(Token::StringLit(s));
            i += 1; // skip closing quote
            continue;
        }

        // Operators: ==, !=, <=, >=, <, >
        if chars[i] == '=' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op(OpKind::Eq));
            i += 2;
            continue;
        }
        if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op(OpKind::Ne));
            i += 2;
            continue;
        }
        if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op(OpKind::Le));
            i += 2;
            continue;
        }
        if chars[i] == '>' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op(OpKind::Ge));
            i += 2;
            continue;
        }
        if chars[i] == '<' {
            tokens.push(Token::Op(OpKind::Lt));
            i += 1;
            continue;
        }
        if chars[i] == '>' {
            tokens.push(Token::Op(OpKind::Gt));
            i += 1;
            continue;
        }

        // Punctuation
        if chars[i] == '(' {
            tokens.push(Token::LParen);
            i += 1;
            continue;
        }
        if chars[i] == ')' {
            tokens.push(Token::RParen);
            i += 1;
            continue;
        }
        if chars[i] == '[' {
            tokens.push(Token::LBracket);
            i += 1;
            continue;
        }
        if chars[i] == ']' {
            tokens.push(Token::RBracket);
            i += 1;
            continue;
        }
        if chars[i] == ',' {
            tokens.push(Token::Comma);
            i += 1;
            continue;
        }

        // Numbers: [+-]? [0-9]* [.]? [0-9]+
        if chars[i].is_ascii_digit()
            || ((chars[i] == '+' || chars[i] == '-')
                && i + 1 < chars.len()
                && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.'))
        {
            let start = i;
            if chars[i] == '+' || chars[i] == '-' {
                i += 1;
            }
            let mut is_float = false;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if i < chars.len() && chars[i] == '.' {
                is_float = true;
                i += 1;
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_str: String = chars[start..i].iter().collect();
            if is_float {
                let v: f64 = num_str
                    .parse()
                    .map_err(|_| EngineError::PredicateParseError {
                        expr: expr.to_string(),
                        reason: format!("invalid float literal: {num_str}"),
                    })?;
                tokens.push(Token::FloatLit(v));
            } else {
                let v: i64 = num_str
                    .parse()
                    .map_err(|_| EngineError::PredicateParseError {
                        expr: expr.to_string(),
                        reason: format!("invalid integer literal: {num_str}"),
                    })?;
                tokens.push(Token::IntLit(v));
            }
            continue;
        }

        // Identifiers and keywords
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let lower = word.to_lowercase();
            match lower.as_str() {
                "and" => tokens.push(Token::And),
                "or" => tokens.push(Token::Or),
                "not" => tokens.push(Token::Not),
                "in" => tokens.push(Token::In),
                "true" => tokens.push(Token::BoolLit(true)),
                "false" => tokens.push(Token::BoolLit(false)),
                _ => tokens.push(Token::Ident(word)),
            }
            continue;
        }

        let reason = if chars[i] == '=' {
            // A lone '=' is the classic "assignment vs equality" slip.
            "unexpected character: '=' (use '==' for equality)".to_string()
        } else {
            format!("unexpected character: '{}'", chars[i])
        };
        return Err(EngineError::PredicateParseError {
            expr: expr.to_string(),
            reason,
        });
    }

    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Parser (A3)
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    schema: &'a Schema,
    expr: String,
    /// "obs" or "var" — used by [`validate_column`] to render the
    /// strsim-augmented "Available {axis} columns: ..." reason on
    /// `EngineError::SchemaError`
    axis: &'a str,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, schema: &'a Schema, expr: &str, axis: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            schema,
            expr: expr.to_string(),
            axis,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(tok)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<()> {
        match self.advance() {
            Some(ref tok) if tok == expected => Ok(()),
            other => Err(EngineError::PredicateParseError {
                expr: self.expr.clone(),
                reason: format!("expected {expected:?}, got {other:?}"),
            }),
        }
    }

    fn parse_error(&self, reason: impl Into<String>) -> EngineError {
        EngineError::PredicateParseError {
            expr: self.expr.clone(),
            reason: reason.into(),
        }
    }

    /// Validate a column exists in the schema and return its data type.
    fn validate_column(&self, col: &str) -> Result<&DataType> {
        match self.schema.column_with_name(col) {
            Some((_, field)) => Ok(field.data_type()),
            None => {
                // Drop pyarrow-internal `__*`
                // columns (notably `__index_level_0__`) from the
                // user-facing available-columns preview.
                let available: Vec<String> = self
                    .schema
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .filter(|n| !n.starts_with("__"))
                    .map(|n| n.to_string())
                    .collect();
                Err(EngineError::SchemaError {
                    column: col.to_string(),
                    reason: crate::index::column_not_found_message(self.axis, col, &available),
                })
            }
        }
    }

    /// Check value–column type compatibility and coerce if necessary.
    ///
    /// A `Dictionary(_, V)` column validates as `V`. Dictionary encoding is
    /// how pandas stores every `Categorical` — of strings, but equally of
    /// integers, floats and bools — so the *value* type is what a literal
    /// has to match. Hardcoding `Utf8` here made an integer-valued
    /// categorical reject both `batch == 1` and `batch == '1'`, leaving the
    /// column unreachable from the query engine.
    fn validate_type(
        &self,
        col: &str,
        col_type: &DataType,
        value: &ScalarValue,
    ) -> Result<ScalarValue> {
        // Look through dictionary encoding and validate against the values.
        if let DataType::Dictionary(_, value_type) = col_type {
            return self.validate_type(col, value_type.as_ref(), value);
        }
        match (col_type, value) {
            // String columns accept string values
            (DataType::Utf8 | DataType::LargeUtf8, ScalarValue::Utf8(_)) => Ok(value.clone()),
            // String column with numeric literal → type mismatch
            (
                DataType::Utf8 | DataType::LargeUtf8,
                ScalarValue::Int64(_) | ScalarValue::Float64(_),
            ) => Err(EngineError::SchemaError {
                column: col.to_string(),
                reason: format!(
                    "type mismatch: column is string but value is {}",
                    match value {
                        ScalarValue::Int64(_) => "integer",
                        ScalarValue::Float64(_) => "float",
                        _ => unreachable!(),
                    }
                ),
            }),
            // Numeric columns accept int/float values
            (
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64,
                ScalarValue::Int64(_) | ScalarValue::Float64(_),
            ) => Ok(value.clone()),
            // Numeric column with a quoted literal → the mirror of the
            // string-column case above. Worth its own arm rather than the
            // catch-all: quoting the value is the first thing a user tries
            // on an integer-valued categorical, and the generic message
            // answers with a raw Arrow `DataType` instead of the two types
            // that actually disagree.
            (
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64,
                ScalarValue::Utf8(_),
            ) => Err(EngineError::SchemaError {
                column: col.to_string(),
                reason: format!(
                    "type mismatch: column is {} but value is a string — drop the quotes",
                    if matches!(col_type, DataType::Float32 | DataType::Float64) {
                        "float"
                    } else {
                        "integer"
                    }
                ),
            }),
            // Boolean column with bool value
            (DataType::Boolean, ScalarValue::Bool(_)) => Ok(value.clone()),
            // Everything else is a type mismatch
            _ => Err(EngineError::SchemaError {
                column: col.to_string(),
                reason: format!(
                    "type mismatch: column type {:?} is not compatible with value {value}",
                    col_type
                ),
            }),
        }
    }

    // --- Recursive descent ---

    fn parse_expr(&mut self) -> Result<Predicate> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> Result<Predicate> {
        let mut left = self.parse_and_expr()?;
        while self.peek() == Some(&Token::Or) {
            self.advance(); // consume 'or'
            let right = self.parse_and_expr()?;
            left = Predicate::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and_expr(&mut self) -> Result<Predicate> {
        let mut left = self.parse_not_expr()?;
        while self.peek() == Some(&Token::And) {
            self.advance(); // consume 'and'
            let right = self.parse_not_expr()?;
            left = Predicate::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not_expr(&mut self) -> Result<Predicate> {
        if self.peek() == Some(&Token::Not) {
            self.advance(); // consume 'not'
            let inner = self.parse_atom()?;
            return Ok(Predicate::Not(Box::new(inner)));
        }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<Predicate> {
        // Parenthesized expression
        if self.peek() == Some(&Token::LParen) {
            self.advance(); // consume '('
            let expr = self.parse_expr()?;
            self.expect(&Token::RParen)?;
            return Ok(expr);
        }

        // Must be an identifier for comparison or in_expr
        let col = match self.advance() {
            Some(Token::Ident(name)) => name,
            other => {
                return Err(self.parse_error(format!("expected identifier, got {other:?}")));
            }
        };

        // Validate column exists
        let col_type = self.validate_column(&col)?.clone();

        // Check next token: operator or 'in'
        match self.peek() {
            Some(Token::In) => {
                self.advance(); // consume 'in'
                self.expect(&Token::LBracket)?;
                let mut values = Vec::new();
                // Parse first value
                let val = self.parse_value()?;
                let val = self.validate_type(&col, &col_type, &val)?;
                values.push(val);
                // Parse remaining comma-separated values
                while self.peek() == Some(&Token::Comma) {
                    self.advance(); // consume ','
                    let val = self.parse_value()?;
                    let val = self.validate_type(&col, &col_type, &val)?;
                    values.push(val);
                }
                self.expect(&Token::RBracket)?;
                Ok(Predicate::In(col, values))
            }
            Some(Token::Op(_)) => {
                let op = match self.advance() {
                    Some(Token::Op(op)) => op,
                    _ => unreachable!(),
                };
                let val = self.parse_value()?;
                let val = self.validate_type(&col, &col_type, &val)?;
                let pred = match op {
                    OpKind::Eq => Predicate::Eq(col, val),
                    OpKind::Ne => Predicate::Ne(col, val),
                    OpKind::Lt => Predicate::Lt(col, val),
                    OpKind::Gt => Predicate::Gt(col, val),
                    OpKind::Le => Predicate::Le(col, val),
                    OpKind::Ge => Predicate::Ge(col, val),
                };
                Ok(pred)
            }
            other => Err(self.parse_error(format!(
                "expected operator or 'in' after column '{col}', got {other:?}"
            ))),
        }
    }

    fn parse_value(&mut self) -> Result<ScalarValue> {
        match self.advance() {
            Some(Token::StringLit(s)) => Ok(ScalarValue::Utf8(s)),
            Some(Token::IntLit(v)) => Ok(ScalarValue::Int64(v)),
            Some(Token::FloatLit(v)) => Ok(ScalarValue::Float64(v)),
            Some(Token::BoolLit(v)) => Ok(ScalarValue::Bool(v)),
            other => Err(self.parse_error(format!("expected value, got {other:?}"))),
        }
    }
}

/// Parse a predicate expression string against an Arrow schema.
///
/// Column names are validated against the schema immediately.
/// Returns `Err(SchemaError)` for unknown columns or type mismatches.
/// `axis` is `"obs"` or `"var"` — used in the strsim-augmented
/// `SchemaError` reason on unknown-column errors (F5-2026-05-20-Tier2).
pub fn parse_predicate(expr: &str, schema: &Schema, axis: &str) -> Result<Predicate> {
    if expr.trim().is_empty() {
        return Err(EngineError::PredicateParseError {
            expr: expr.to_string(),
            reason: "empty predicate expression".to_string(),
        });
    }

    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens, schema, expr, axis);
    let pred = parser.parse_expr()?;

    // Ensure all tokens were consumed
    if parser.pos < parser.tokens.len() {
        return Err(EngineError::PredicateParseError {
            expr: expr.to_string(),
            reason: format!(
                "unexpected token after end of expression: {:?}",
                parser.tokens[parser.pos]
            ),
        });
    }

    Ok(pred)
}

// ---------------------------------------------------------------------------
// Evaluator (A4)
// ---------------------------------------------------------------------------

/// Evaluate a predicate against an Arrow RecordBatch, returning a boolean mask.
///
/// **Null semantics: three-valued (Kleene) logic, coalesced to `false` at the
/// top level** — i.e. a SQL `WHERE` clause. A comparison against a NULL cell is
/// UNKNOWN, not `false`; `and` / `or` combine UNKNOWN per Kleene
/// (`null OR true = true`, `null AND false = false`), `not` propagates it, and
/// only the final mask turns a surviving UNKNOWN into "not matched". pandas,
/// polars and SQL agree on `or`; see `docs/api.md` § QueryPipeline.
///
/// Getting this wrong is silent: combining with arrow's *non*-Kleene
/// `boolean::or` makes `null OR true` null, and the top-level coalesce then
/// drops a row that every other engine returns.
pub fn evaluate(predicate: &Predicate, batch: &RecordBatch) -> Result<BooleanArray> {
    let mask = eval_inner(predicate, batch)?;
    // Coalesce nulls to false at the top level
    if mask.null_count() > 0 {
        let result: BooleanArray = (0..mask.len())
            .map(|i| {
                Some(if mask.is_null(i) {
                    false
                } else {
                    mask.value(i)
                })
            })
            .collect();
        Ok(result)
    } else {
        Ok(mask)
    }
}

/// Evaluate leaves as Arrow comparison kernels over a decoded batch, keeping
/// null identity (UNKNOWN) intact through the tree.
struct MaskAlgebra<'a> {
    batch: &'a RecordBatch,
}

impl PredicateAlgebra for MaskAlgebra<'_> {
    type Value = BooleanArray;

    fn leaf(&self, pred: &Predicate) -> Result<Self::Value> {
        let batch = self.batch;
        match pred {
            Predicate::Eq(col, val) => eval_comparison(batch, col, val, CmpOp::Eq),
            Predicate::Ne(col, val) => eval_comparison(batch, col, val, CmpOp::Ne),
            Predicate::Lt(col, val) => eval_comparison(batch, col, val, CmpOp::Lt),
            Predicate::Gt(col, val) => eval_comparison(batch, col, val, CmpOp::Gt),
            Predicate::Le(col, val) => eval_comparison(batch, col, val, CmpOp::Le),
            Predicate::Ge(col, val) => eval_comparison(batch, col, val, CmpOp::Ge),
            Predicate::In(col, vals) => {
                // OR of individual Eq comparisons. Every operand compares the
                // SAME column, and the comparison helpers propagate null purely
                // from the input cell (independent of the scalar), so all
                // operands share one null pattern and the Kleene and non-Kleene
                // kernels agree here. `or_kleene` is used anyway so the whole
                // evaluator has one rule.
                let mut result: Option<BooleanArray> = None;
                for val in vals {
                    let mask = eval_comparison(batch, col, val, CmpOp::Eq)?;
                    result = Some(match result {
                        None => mask,
                        Some(prev) => compute::kernels::boolean::or_kleene(&prev, &mask)?,
                    });
                }
                // If vals is empty, return all-false
                Ok(result.unwrap_or_else(|| BooleanArray::from(vec![false; batch.num_rows()])))
            }
            // `walk` handles these and never routes them here.
            Predicate::And(_, _) | Predicate::Or(_, _) | Predicate::Not(_) => {
                unreachable!("walk dispatches combinators")
            }
        }
    }

    fn and(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value> {
        // `and_kleene`, not `and`: `null AND false` is FALSE (no value of the
        // unknown operand makes the conjunction true). At the top level that is
        // indistinguishable from arrow's non-Kleene `and` — both land on "not
        // matched" — but under a `Not` it is the difference between
        // `not (null AND false)` being true (correct) and false.
        Ok(compute::kernels::boolean::and_kleene(&a, &b)?)
    }

    fn or(&self, a: Self::Value, b: Self::Value) -> Result<Self::Value> {
        // `or_kleene`, not `or`: `null OR true` is TRUE. Arrow's non-Kleene
        // `or` returns null whenever either side is null, which the top-level
        // coalesce turns into `false` — silently dropping a row whose other
        // operand matched.
        Ok(compute::kernels::boolean::or_kleene(&a, &b)?)
    }

    fn not(&self, a: Self::Value) -> Result<Self::Value> {
        // Arrow's `not` is already Kleene-correct: null stays null (NOT UNKNOWN
        // is UNKNOWN), true→false, false→true.
        Ok(compute::kernels::boolean::not(&a)?)
    }
}

/// Internal evaluator that preserves null identity through the expression tree.
fn eval_inner(predicate: &Predicate, batch: &RecordBatch) -> Result<BooleanArray> {
    walk(predicate, &MaskAlgebra { batch })
}

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

/// Evaluate a single column comparison against a scalar value.
fn eval_comparison(
    batch: &RecordBatch,
    col_name: &str,
    value: &ScalarValue,
    op: CmpOp,
) -> Result<BooleanArray> {
    let col_idx = batch
        .schema()
        .index_of(col_name)
        .map_err(EngineError::ArrowError)?;
    eval_comparison_on_array(col_name, batch.column(col_idx), value, op)
}

/// The type-dispatch half of [`eval_comparison`], over an array rather than
/// a batch column. Split out so the dictionary arm can decode to its value
/// array and re-enter here; `col_name` is carried only for error messages.
fn eval_comparison_on_array(
    col_name: &str,
    column: &ArrayRef,
    value: &ScalarValue,
    op: CmpOp,
) -> Result<BooleanArray> {
    let dtype = column.data_type();

    match dtype {
        DataType::Utf8 => eval_utf8(column.as_string::<i32>(), value, op),
        DataType::LargeUtf8 => eval_utf8(column.as_string::<i64>(), value, op),
        DataType::Int8 => {
            eval_numeric_array::<arrow::datatypes::Int8Type>(column.as_primitive(), value, op)
        }
        DataType::Int16 => {
            eval_numeric_array::<arrow::datatypes::Int16Type>(column.as_primitive(), value, op)
        }
        DataType::Int32 => {
            eval_numeric_array::<arrow::datatypes::Int32Type>(column.as_primitive(), value, op)
        }
        DataType::Int64 => {
            eval_numeric_array::<arrow::datatypes::Int64Type>(column.as_primitive(), value, op)
        }
        DataType::UInt8 => {
            eval_numeric_array::<arrow::datatypes::UInt8Type>(column.as_primitive(), value, op)
        }
        DataType::UInt16 => {
            eval_numeric_array::<arrow::datatypes::UInt16Type>(column.as_primitive(), value, op)
        }
        DataType::UInt32 => {
            eval_numeric_array::<arrow::datatypes::UInt32Type>(column.as_primitive(), value, op)
        }
        DataType::UInt64 => {
            eval_numeric_array::<arrow::datatypes::UInt64Type>(column.as_primitive(), value, op)
        }
        DataType::Float32 => {
            eval_numeric_array::<arrow::datatypes::Float32Type>(column.as_primitive(), value, op)
        }
        DataType::Float64 => {
            eval_numeric_array::<arrow::datatypes::Float64Type>(column.as_primitive(), value, op)
        }
        DataType::Boolean => eval_boolean(column.as_boolean(), value, op),
        DataType::Dictionary(_, value_type) => match value_type.as_ref() {
            // Strings keep the zero-decode path: compare against the (small)
            // dictionary values and gather through the keys.
            DataType::Utf8 | DataType::LargeUtf8 => eval_dictionary_utf8(column, value, op),
            // Any other value type — an integer / float / bool pandas
            // `Categorical` — decodes to its value array and re-enters the
            // primitive arms above, so a dictionary-encoded column compares
            // exactly like the plain column it holds. Decoding
            // costs one pass; `Predicate::In` is an OR of `Eq` and so decodes
            // once per member, which is fine at categorical cardinality.
            _ => {
                let decoded = arrow::compute::cast(column, value_type.as_ref()).map_err(|e| {
                    EngineError::SchemaError {
                        column: col_name.to_string(),
                        reason: format!(
                            "could not decode dictionary column with value type \
                                 {value_type:?}: {e}"
                        ),
                    }
                })?;
                eval_comparison_on_array(col_name, &decoded, value, op)
            }
        },
        _ => Err(EngineError::SchemaError {
            column: col_name.to_string(),
            reason: format!("unsupported column type: {dtype:?}"),
        }),
    }
}

/// Evaluate comparison on a string (Utf8 / LargeUtf8) array.
fn eval_utf8<O: arrow::array::OffsetSizeTrait>(
    array: &arrow::array::GenericStringArray<O>,
    value: &ScalarValue,
    op: CmpOp,
) -> Result<BooleanArray> {
    let ScalarValue::Utf8(ref s) = value else {
        return Err(EngineError::SchemaError {
            column: String::new(),
            reason: format!("expected Utf8 value for string column, got {value}"),
        });
    };
    let result: BooleanArray = (0..array.len())
        .map(|i| {
            if array.is_null(i) {
                None // null → propagate null
            } else {
                let v = array.value(i);
                Some(match op {
                    CmpOp::Eq => v == s.as_str(),
                    CmpOp::Ne => v != s.as_str(),
                    CmpOp::Lt => v < s.as_str(),
                    CmpOp::Gt => v > s.as_str(),
                    CmpOp::Le => v <= s.as_str(),
                    CmpOp::Ge => v >= s.as_str(),
                })
            }
        })
        .collect();
    Ok(result)
}

/// Evaluate comparison on any numeric PrimitiveArray by promoting values to f64.
fn eval_numeric_array<T>(
    array: &arrow::array::PrimitiveArray<T>,
    value: &ScalarValue,
    op: CmpOp,
) -> Result<BooleanArray>
where
    T: arrow::datatypes::ArrowPrimitiveType,
    T::Native: arrow::datatypes::ArrowNativeTypeOp,
{
    // For integer comparisons, use i128 when possible to avoid f64 precision loss
    // for values > 2^53. Fall back to f64 for float comparisons.
    let cmp_i128: Option<i128> = match value {
        ScalarValue::Int64(v) => Some(*v as i128),
        _ => None,
    };
    let cmp_val: f64 = match value {
        ScalarValue::Int64(v) => *v as f64,
        ScalarValue::Float64(v) => *v,
        _ => {
            return Err(EngineError::SchemaError {
                column: String::new(),
                reason: format!("expected numeric value for numeric column, got {value}"),
            })
        }
    };

    let result: BooleanArray = (0..array.len())
        .map(|i| {
            if array.is_null(i) {
                None // null → propagate null
            } else {
                let native = array.value(i);
                // If both sides are integers, compare via i128 to avoid f64
                // precision loss for values > 2^53.
                if let Some(cmp_int) = cmp_i128 {
                    if let Some(native_int) = native_to_i128(native) {
                        return Some(match op {
                            CmpOp::Eq => native_int == cmp_int,
                            CmpOp::Ne => native_int != cmp_int,
                            CmpOp::Lt => native_int < cmp_int,
                            CmpOp::Gt => native_int > cmp_int,
                            CmpOp::Le => native_int <= cmp_int,
                            CmpOp::Ge => native_int >= cmp_int,
                        });
                    }
                }
                // Fall back to f64 comparison for float columns or unrecognized types.
                // Unrecognized native types yield None (treated as null)
                // rather than silently returning 0.0 which would give wrong results.
                native_to_f64(native).map(|v| match op {
                    CmpOp::Eq => v == cmp_val,
                    CmpOp::Ne => v != cmp_val,
                    CmpOp::Lt => v < cmp_val,
                    CmpOp::Gt => v > cmp_val,
                    CmpOp::Le => v <= cmp_val,
                    CmpOp::Ge => v >= cmp_val,
                })
            }
        })
        .collect();
    Ok(result)
}

/// Convert any ArrowNativeType to f64 via monomorphized type dispatch.
/// This handles all integer types (i8–i64, u8–u64) and float types (f32, f64).
/// Returns `None` for unrecognized types rather than silently returning 0.0,
/// which would produce wrong query results.
fn native_to_f64<N: arrow::datatypes::ArrowNativeType>(v: N) -> Option<f64> {
    // The downcast_ref chain below looks like runtime type dispatch, but
    // it is fully optimized away by LLVM. Since eval_numeric_array is
    // generic over T: ArrowPrimitiveType, each call site monomorphizes
    // this function for a concrete N. LLVM resolves all TypeId comparisons
    // at compile time and eliminates dead branches — e.g., native_to_f64::<i32>
    // compiles to a single cvtsi2sd instruction.
    use std::any::Any;
    let any_ref: &dyn Any = &v;
    if let Some(&val) = any_ref.downcast_ref::<i8>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<i16>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<i32>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<i64>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<u8>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<u16>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<u32>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<u64>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<f32>() {
        return Some(val as f64);
    }
    if let Some(&val) = any_ref.downcast_ref::<f64>() {
        return Some(val);
    }
    // Unrecognized type (e.g. half::f16) — return None so callers
    // treat this as null rather than silently using 0.0
    None
}

/// Convert an integer ArrowNativeType to i128 for lossless integer comparison.
/// Returns `None` for float types or unrecognized types — callers should fall
/// back to f64 comparison in that case.
///
/// Like `native_to_f64`, the downcast_ref chain is fully optimized away by
/// LLVM via monomorphization — see that function's doc comment for details.
fn native_to_i128<N: arrow::datatypes::ArrowNativeType>(v: N) -> Option<i128> {
    use std::any::Any;
    let any_ref: &dyn Any = &v;
    if let Some(&val) = any_ref.downcast_ref::<i8>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<i16>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<i32>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<i64>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<u8>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<u16>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<u32>() {
        return Some(val as i128);
    }
    if let Some(&val) = any_ref.downcast_ref::<u64>() {
        return Some(val as i128);
    }
    // Float types and unrecognized types — not integer, return None
    None
}

/// Evaluate comparison on a boolean array.
fn eval_boolean(array: &BooleanArray, value: &ScalarValue, op: CmpOp) -> Result<BooleanArray> {
    let ScalarValue::Bool(cmp_val) = value else {
        return Err(EngineError::SchemaError {
            column: String::new(),
            reason: format!("expected boolean value for boolean column, got {value}"),
        });
    };

    let result: BooleanArray = (0..array.len())
        .map(|i| {
            if array.is_null(i) {
                None // null → propagate null
            } else {
                let v = array.value(i);
                Some(match op {
                    CmpOp::Eq => v == *cmp_val,
                    CmpOp::Ne => v != *cmp_val,
                    // Booleans: false < true
                    CmpOp::Lt => !v && *cmp_val,
                    CmpOp::Gt => v && !*cmp_val,
                    CmpOp::Le => v <= *cmp_val,
                    CmpOp::Ge => v >= *cmp_val,
                })
            }
        })
        .collect();
    Ok(result)
}

/// Evaluate comparison on a dictionary-encoded column.
/// Resolves comparison against dictionary values, then maps to indices.
fn eval_dictionary_utf8(
    column: &dyn Array,
    value: &ScalarValue,
    op: CmpOp,
) -> Result<BooleanArray> {
    let ScalarValue::Utf8(ref s) = value else {
        return Err(EngineError::SchemaError {
            column: String::new(),
            reason: format!("expected Utf8 value for dictionary column, got {value}"),
        });
    };

    // Try common dictionary key types: Int8, Int16, Int32
    // Arrow's DictionaryArray is parameterized by key type
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int8Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int16Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int64Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt8Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt16Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }
    if let Some(dict) = column
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::UInt32Type>>()
    {
        return eval_dict_typed(dict, s, op);
    }

    Err(EngineError::SchemaError {
        column: String::new(),
        reason: "unsupported dictionary key type".to_string(),
    })
}

/// Evaluate string comparison on a typed dictionary array by looking up
/// values from the dictionary and comparing.
fn eval_dict_typed<K>(
    dict: &arrow::array::DictionaryArray<K>,
    cmp_str: &str,
    op: CmpOp,
) -> Result<BooleanArray>
where
    K: arrow::datatypes::ArrowDictionaryKeyType,
    K::Native: TryInto<usize> + Copy,
{
    let values = dict.values();
    // Get string values from the dictionary
    let str_values = values
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| EngineError::SchemaError {
            column: String::new(),
            reason: "dictionary values are not Utf8".to_string(),
        })?;

    let result: BooleanArray = (0..dict.len())
        .map(|i| {
            if dict.is_null(i) {
                None // null → propagate null
            } else {
                let keys = dict.keys();
                let key_val = keys.value(i);
                let idx: usize = match key_val.try_into() {
                    Ok(v) => v,
                    Err(_) => return Some(false),
                };
                if idx >= str_values.len() || str_values.is_null(idx) {
                    return Some(false);
                }
                let v = str_values.value(idx);
                Some(match op {
                    CmpOp::Eq => v == cmp_str,
                    CmpOp::Ne => v != cmp_str,
                    CmpOp::Lt => v < cmp_str,
                    CmpOp::Gt => v > cmp_str,
                    CmpOp::Le => v <= cmp_str,
                    CmpOp::Ge => v >= cmp_str,
                })
            }
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, DictionaryArray, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use std::sync::Arc;

    /// Helper: build a schema for tests.
    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("cell_type", DataType::Utf8, true),
            Field::new("n_genes", DataType::Int64, false),
            Field::new("n_counts", DataType::Int64, false),
            Field::new("is_doublet", DataType::Boolean, true),
            Field::new("tissue", DataType::Utf8, true),
            Field::new("disease", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ])
    }

    // -----------------------------------------------------------------------
    // Parser tests (A3)
    // -----------------------------------------------------------------------

    #[test]
    fn parse_equality() {
        let schema = test_schema();
        let pred = parse_predicate("cell_type == 'T cell'", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()))
        );
    }

    #[test]
    fn parse_double_quoted_string() {
        let schema = test_schema();
        let pred = parse_predicate("cell_type == \"T cell\"", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()))
        );
    }

    #[test]
    fn parse_and_comparison() {
        let schema = test_schema();
        let pred = parse_predicate("n_genes > 200 and n_counts < 5000", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::And(
                Box::new(Predicate::Gt("n_genes".into(), ScalarValue::Int64(200))),
                Box::new(Predicate::Lt("n_counts".into(), ScalarValue::Int64(5000))),
            )
        );
    }

    #[test]
    fn parse_in_expr() {
        let schema = test_schema();
        let pred = parse_predicate(
            "cell_type in ['T cell', 'B cell', 'NK cell']",
            &schema,
            "obs",
        )
        .unwrap();
        assert_eq!(
            pred,
            Predicate::In(
                "cell_type".into(),
                vec![
                    ScalarValue::Utf8("T cell".into()),
                    ScalarValue::Utf8("B cell".into()),
                    ScalarValue::Utf8("NK cell".into()),
                ]
            )
        );
    }

    #[test]
    fn parse_not() {
        let schema = test_schema();
        let pred = parse_predicate("not is_doublet == true", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::Not(Box::new(Predicate::Eq(
                "is_doublet".into(),
                ScalarValue::Bool(true)
            )))
        );
    }

    #[test]
    fn parse_precedence_parens() {
        let schema = test_schema();
        let pred = parse_predicate(
            "(tissue == 'lung' or tissue == 'heart') and disease == 'healthy'",
            &schema,
            "obs",
        )
        .unwrap();
        assert_eq!(
            pred,
            Predicate::And(
                Box::new(Predicate::Or(
                    Box::new(Predicate::Eq(
                        "tissue".into(),
                        ScalarValue::Utf8("lung".into())
                    )),
                    Box::new(Predicate::Eq(
                        "tissue".into(),
                        ScalarValue::Utf8("heart".into())
                    )),
                )),
                Box::new(Predicate::Eq(
                    "disease".into(),
                    ScalarValue::Utf8("healthy".into())
                )),
            )
        );
    }

    #[test]
    fn parse_or_and_precedence() {
        // "a or b and c" should parse as "a or (b and c)"
        let schema = test_schema();
        let pred = parse_predicate(
            "tissue == 'lung' or tissue == 'heart' and disease == 'healthy'",
            &schema,
            "obs",
        )
        .unwrap();
        // tissue == 'lung' OR (tissue == 'heart' AND disease == 'healthy')
        assert_eq!(
            pred,
            Predicate::Or(
                Box::new(Predicate::Eq(
                    "tissue".into(),
                    ScalarValue::Utf8("lung".into())
                )),
                Box::new(Predicate::And(
                    Box::new(Predicate::Eq(
                        "tissue".into(),
                        ScalarValue::Utf8("heart".into())
                    )),
                    Box::new(Predicate::Eq(
                        "disease".into(),
                        ScalarValue::Utf8("healthy".into())
                    )),
                )),
            )
        );
    }

    #[test]
    fn parse_nonexistent_column_error() {
        let schema = test_schema();
        let err = parse_predicate("nonexistent_column == 'x'", &schema, "obs").unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
    }

    // F5-2026-05-20-Tier2: runtime predicate path should produce the
    // same strsim-augmented "Available / Did you mean" suffix that
    // convert-time forced-index errors got from PR #113.
    #[test]
    fn parse_unknown_column_includes_available_and_suggestion() {
        let schema = test_schema();
        let err = parse_predicate("celltype == 'T cell'", &schema, "obs").unwrap_err();
        match err {
            EngineError::SchemaError { column, reason } => {
                assert_eq!(column, "celltype");
                assert!(
                    reason.contains("column not found"),
                    "reason should start with 'column not found': {reason}"
                );
                assert!(
                    reason.contains("Available obs columns"),
                    "reason should list available columns: {reason}"
                );
                assert!(
                    reason.contains("Did you mean 'cell_type'?"),
                    "reason should propose 'cell_type' (strsim ≥ 0.6): {reason}"
                );
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn parse_unknown_column_no_suggestion_when_far() {
        let schema = test_schema();
        let err = parse_predicate("xyz_completely_unrelated == 1", &schema, "obs").unwrap_err();
        match err {
            EngineError::SchemaError { reason, .. } => {
                assert!(
                    reason.contains("column not found"),
                    "reason should start with 'column not found': {reason}"
                );
                assert!(
                    reason.contains("Available obs columns"),
                    "reason should list available columns: {reason}"
                );
                assert!(
                    !reason.contains("Did you mean"),
                    "no near match should appear: {reason}"
                );
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn parse_unknown_var_column_uses_var_axis_label() {
        // Verify the axis label flows through from filter_var-style calls.
        let schema = test_schema();
        let err = parse_predicate("nonsense_col == 'x'", &schema, "var").unwrap_err();
        match err {
            EngineError::SchemaError { reason, .. } => {
                assert!(
                    reason.contains("Available var columns"),
                    "reason should label axis as 'var': {reason}"
                );
            }
            other => panic!("expected SchemaError, got {other:?}"),
        }
    }

    #[test]
    fn parse_type_mismatch_string_vs_int() {
        let schema = test_schema();
        let err = parse_predicate("cell_type > 5", &schema, "obs").unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
    }

    #[test]
    fn parse_empty_string_error() {
        let schema = test_schema();
        let err = parse_predicate("", &schema, "obs").unwrap_err();
        assert!(matches!(err, EngineError::PredicateParseError { .. }));
    }

    #[test]
    fn parse_malformed_error() {
        let schema = test_schema();
        let err = parse_predicate("cell_type ==", &schema, "obs").unwrap_err();
        assert!(matches!(err, EngineError::PredicateParseError { .. }));
    }

    #[test]
    fn parse_lone_equals_hints_double_equals() {
        let schema = test_schema();
        let err = parse_predicate("cell_type = 'T cell'", &schema, "obs").unwrap_err();
        match err {
            EngineError::PredicateParseError { reason, .. } => {
                assert!(
                    reason.contains("use '=='"),
                    "lone '=' should hint at '==', got: {reason}"
                );
            }
            other => panic!("expected PredicateParseError, got {other:?}"),
        }
    }

    #[test]
    fn parse_float_value() {
        let schema = test_schema();
        let pred = parse_predicate("score > 0.5", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::Gt("score".into(), ScalarValue::Float64(0.5))
        );
    }

    #[test]
    fn parse_le_ge() {
        let schema = test_schema();
        let pred = parse_predicate("n_genes >= 100 and n_genes <= 5000", &schema, "obs").unwrap();
        assert_eq!(
            pred,
            Predicate::And(
                Box::new(Predicate::Ge("n_genes".into(), ScalarValue::Int64(100))),
                Box::new(Predicate::Le("n_genes".into(), ScalarValue::Int64(5000))),
            )
        );
    }

    #[test]
    fn predicate_columns() {
        let pred = Predicate::And(
            Box::new(Predicate::Eq(
                "cell_type".into(),
                ScalarValue::Utf8("T".into()),
            )),
            Box::new(Predicate::Gt("n_genes".into(), ScalarValue::Int64(200))),
        );
        let mut cols = pred.columns();
        cols.sort();
        assert_eq!(cols, vec!["cell_type", "n_genes"]);
    }

    #[test]
    fn predicate_display() {
        let pred = Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()));
        assert_eq!(pred.to_string(), "cell_type == 'T cell'");
    }

    // -----------------------------------------------------------------------
    // Evaluator tests (A4)
    // -----------------------------------------------------------------------

    fn test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, false),
            Field::new("color", DataType::Utf8, true),
            Field::new("active", DataType::Boolean, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![
                    Some("Alice"),
                    Some("Bob"),
                    None,
                    Some("Diana"),
                    Some("Eve"),
                ])),
                Arc::new(Int64Array::from(vec![25, 35, 40, 45, 20])),
                Arc::new(StringArray::from(vec![
                    Some("red"),
                    Some("blue"),
                    Some("red"),
                    Some("green"),
                    Some("blue"),
                ])),
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    Some(false),
                    None,
                    Some(true),
                    Some(false),
                ])),
            ],
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Dictionary value types other than string
    // -----------------------------------------------------------------------

    /// `pd.Categorical([...])` over integers / bools / floats, keyed
    /// `Int32` as `widen_dictionary_keys` normalises them on disk.
    fn dict_batch() -> RecordBatch {
        use arrow::array::Float64Array;

        let batch_keys = Int32Array::from(vec![0, 1, 2, 0, 1]);
        let batch_values: ArrayRef = Arc::new(Int64Array::from(vec![10i64, 20, 30]));
        let batch_col: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(batch_keys, batch_values).unwrap());

        let flag_keys = Int32Array::from(vec![Some(0), Some(1), None, Some(0), Some(1)]);
        let flag_values: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let flag_col: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(flag_keys, flag_values).unwrap());

        let score_keys = Int32Array::from(vec![0, 1, 0, 1, 0]);
        let score_values: ArrayRef = Arc::new(Float64Array::from(vec![1.5f64, 2.5]));
        let score_col: ArrayRef =
            Arc::new(DictionaryArray::<Int32Type>::try_new(score_keys, score_values).unwrap());

        let schema = Arc::new(Schema::new(vec![
            Field::new("batch", batch_col.data_type().clone(), true),
            Field::new("flag", flag_col.data_type().clone(), true),
            Field::new("score", score_col.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![batch_col, flag_col, score_col]).unwrap()
    }

    fn dict_schema() -> Schema {
        (*dict_batch().schema()).clone()
    }

    /// The core case: `batch == 20` must parse **and** select the
    /// right rows. A test that only asserts `parse_predicate(...).is_ok()`
    /// would pass against a version that returns every row, or none.
    #[test]
    fn integer_categorical_accepts_an_integer_literal() {
        let schema = dict_schema();
        let pred = parse_predicate("batch == 20", &schema, "obs").expect("must parse");
        assert_eq!(pred, Predicate::Eq("batch".into(), ScalarValue::Int64(20)));
        let mask = evaluate(&pred, &dict_batch()).unwrap();
        // dictionary values [10, 20, 30], keys [0, 1, 2, 0, 1]
        let expected = [false, true, false, false, true];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    /// Ordering operators have to work too — the numeric arm is reached
    /// only after the dictionary is decoded to its value type.
    #[test]
    fn integer_categorical_supports_ordering_and_in() {
        let schema = dict_schema();
        let batch = dict_batch();

        let gt = parse_predicate("batch > 15", &schema, "obs").unwrap();
        let mask = evaluate(&gt, &batch).unwrap();
        for (i, &exp) in [false, true, true, false, true].iter().enumerate() {
            assert_eq!(mask.value(i), exp, "gt row {i}");
        }

        let in_pred = parse_predicate("batch in [10, 30]", &schema, "obs").unwrap();
        let mask = evaluate(&in_pred, &batch).unwrap();
        for (i, &exp) in [true, false, true, true, false].iter().enumerate() {
            assert_eq!(mask.value(i), exp, "in row {i}");
        }
    }

    /// A float-valued categorical is the same shape one type over.
    #[test]
    fn float_categorical_accepts_a_float_literal() {
        let schema = dict_schema();
        let pred = parse_predicate("score > 2.0", &schema, "obs").unwrap();
        let mask = evaluate(&pred, &dict_batch()).unwrap();
        for (i, &exp) in [false, true, false, true, false].iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    /// A boolean-valued categorical must behave exactly as the plain
    /// `Boolean` column it decodes to, nulls included.
    #[test]
    fn boolean_categorical_evaluates_like_a_plain_bool() {
        let schema = dict_schema();
        let pred = parse_predicate("flag == true", &schema, "obs").unwrap();
        let mask = evaluate(&pred, &dict_batch()).unwrap();
        // keys [0, 1, null, 0, 1] over values [true, false];
        // the null key coalesces to false at the top level.
        for (i, &exp) in [true, false, false, true, false].iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    /// The mirror of the existing "column is string but value is
    /// integer" message. This is the error a user meets after guessing
    /// the wrong syntax for an integer categorical, so it has to name
    /// both sides rather than print a raw Arrow `DataType`.
    #[test]
    fn string_literal_against_a_numeric_column_names_both_types() {
        let schema = test_schema();
        let err = parse_predicate("n_genes == '200'", &schema, "obs").unwrap_err();
        let msg = err.to_string();
        let low = msg.to_lowercase();
        assert!(
            low.contains("integer") && low.contains("string"),
            "expected both types named, got: {msg}"
        );
        assert!(
            !msg.contains("Int64"),
            "should not leak the raw Arrow dtype: {msg}"
        );
    }

    /// And the same for an integer-valued categorical, which is the
    /// column an analyst actually has.
    #[test]
    fn string_literal_against_an_integer_categorical_is_rejected() {
        let schema = dict_schema();
        let err = parse_predicate("batch == '20'", &schema, "obs").unwrap_err();
        let low = err.to_string().to_lowercase();
        assert!(
            low.contains("integer") && low.contains("string"),
            "expected both types named, got: {err}"
        );
    }

    #[test]
    fn eval_eq_string() {
        let batch = test_batch();
        let pred = Predicate::Eq("name".into(), ScalarValue::Utf8("Alice".into()));
        let mask = evaluate(&pred, &batch).unwrap();
        let expected: Vec<bool> = vec![true, false, false, false, false];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_range() {
        let batch = test_batch();
        let pred = Predicate::And(
            Box::new(Predicate::Gt("age".into(), ScalarValue::Int64(30))),
            Box::new(Predicate::Lt("age".into(), ScalarValue::Int64(50))),
        );
        let mask = evaluate(&pred, &batch).unwrap();
        // ages: 25, 35, 40, 45, 20 → matches: false, true, true, true, false
        let expected = [false, true, true, true, false];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_in() {
        let batch = test_batch();
        let pred = Predicate::In(
            "color".into(),
            vec![
                ScalarValue::Utf8("red".into()),
                ScalarValue::Utf8("blue".into()),
            ],
        );
        let mask = evaluate(&pred, &batch).unwrap();
        // colors: red, blue, red, green, blue → matches: t, t, t, f, t
        let expected = [true, true, true, false, true];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_null_produces_false() {
        let batch = test_batch();
        // name column has null at index 2
        let pred = Predicate::Eq("name".into(), ScalarValue::Utf8("Alice".into()));
        let mask = evaluate(&pred, &batch).unwrap();
        assert!(!mask.value(2)); // null → false
    }

    #[test]
    fn eval_or_with_null_operand_matches_sql_semantics() {
        // `name` is NULL at row 2, whose `age` is 40:
        //
        //   name == 'Alice'  ->  [true,  false, null, false, false]
        //   age  >  30       ->  [false, true,  true, true,  false]
        //
        // In three-valued logic `null OR true` is TRUE — no assignment of the
        // unknown left operand can make the disjunction false — so row 2
        // matches. pandas, polars, scanpy and SQL all return it.
        let batch = test_batch();
        let pred = Predicate::Or(
            Box::new(Predicate::Eq(
                "name".into(),
                ScalarValue::Utf8("Alice".into()),
            )),
            Box::new(Predicate::Gt("age".into(), ScalarValue::Int64(30))),
        );
        let mask = evaluate(&pred, &batch).unwrap();
        let got: Vec<bool> = (0..mask.len()).map(|i| mask.value(i)).collect();
        assert_eq!(got, vec![true, true, true, true, false]);
        assert!(
            mask.value(2),
            "row 2 has a NULL `name` and age=40; `null OR true` is true, not false"
        );
    }

    #[test]
    fn eval_or_with_nulls_is_operand_order_independent() {
        let batch = test_batch();
        let a = Predicate::Eq("name".into(), ScalarValue::Utf8("Alice".into()));
        let b = Predicate::Gt("age".into(), ScalarValue::Int64(30));
        let collect = |p: &Predicate| -> Vec<bool> {
            let m = evaluate(p, &batch).unwrap();
            (0..m.len()).map(|i| m.value(i)).collect()
        };
        assert_eq!(
            collect(&Predicate::Or(Box::new(a.clone()), Box::new(b.clone()))),
            collect(&Predicate::Or(Box::new(b), Box::new(a))),
        );
    }

    #[test]
    fn eval_not_of_and_with_null_operand_matches_sql_semantics() {
        // The `and` half of the same fix. At the top level Kleene and
        // non-Kleene `AND` agree (a null conjunct lands on false either way),
        // so only a negated `AND` can tell them apart:
        //
        //   name == 'Alice'  ->  [true, false, null,  false, false]
        //   age  <  30       ->  [true, false, false, false, true ]
        //
        // `null AND false` is FALSE in three-valued logic, so `not (...)` is
        // TRUE at row 2.
        let batch = test_batch();
        let pred = Predicate::Not(Box::new(Predicate::And(
            Box::new(Predicate::Eq(
                "name".into(),
                ScalarValue::Utf8("Alice".into()),
            )),
            Box::new(Predicate::Lt("age".into(), ScalarValue::Int64(30))),
        )));
        let mask = evaluate(&pred, &batch).unwrap();
        let expected = [false, true, true, true, true];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_not() {
        let batch = test_batch();
        let pred = Predicate::Not(Box::new(Predicate::Eq(
            "name".into(),
            ScalarValue::Utf8("Alice".into()),
        )));
        let mask = evaluate(&pred, &batch).unwrap();
        // name: Alice, Bob, null, Diana, Eve
        // NOT Eq("Alice"): false, true, false (null→false), true, true
        let expected = [false, true, false, true, true];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_dictionary_column() {
        // Build a dictionary-encoded column
        let keys = Int32Array::from(vec![0, 1, 0, 2, 1]);
        let values = StringArray::from(vec!["T cell", "B cell", "NK cell"]);
        let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "cell_type",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dict)]).unwrap();

        let pred = Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()));
        let mask = evaluate(&pred, &batch).unwrap();
        // indices: 0,1,0,2,1 → values: T cell, B cell, T cell, NK cell, B cell
        let expected = [true, false, true, false, false];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    #[test]
    fn eval_boolean_column() {
        let batch = test_batch();
        let pred = Predicate::Eq("active".into(), ScalarValue::Bool(true));
        let mask = evaluate(&pred, &batch).unwrap();
        // active: true, false, null, true, false → matches: t, f, f (null), t, f
        let expected = [true, false, false, true, false];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(mask.value(i), exp, "row {i}");
        }
    }

    // -----------------------------------------------------------------------
    // Row-set evaluator tests
    // -----------------------------------------------------------------------
    mod rowset_eval {
        use crate::index::{
            CategoricalEntry, CategoricalIndex, IndexedColumn, PredicateIndex, ShardRange,
        };
        use crate::predicate::{eval_rowset, Predicate, RowSetCtx, ScalarValue};
        use crate::pushdown::CategoryDictionaries;
        use crate::rowset::RowRange;

        fn sr(shard_id: u32, row_start: u32, row_end: u32) -> ShardRange {
            ShardRange {
                shard_id,
                row_start,
                row_end,
            }
        }

        fn idx() -> PredicateIndex {
            // obs shards: 0 -> [0,10), 1 -> [10,20)
            // cell_type: "B cell" in shard0 rows[0,5), shard1 rows[0,3) (global 10..13)
            //            "T cell" in shard0 rows[5,10)
            // tissue:    "blood"  in shard0 rows[0,10)
            PredicateIndex {
                version: 1,
                columns: vec![
                    IndexedColumn::Categorical(CategoricalIndex {
                        column_name: "cell_type".into(),
                        entries: vec![
                            CategoricalEntry {
                                value: "B cell".into(),
                                shard_ranges: vec![sr(0, 0, 5), sr(1, 0, 3)],
                            },
                            CategoricalEntry {
                                value: "T cell".into(),
                                shard_ranges: vec![sr(0, 5, 10)],
                            },
                        ],
                    }),
                    IndexedColumn::Categorical(CategoricalIndex {
                        column_name: "tissue".into(),
                        entries: vec![CategoricalEntry {
                            value: "blood".into(),
                            shard_ranges: vec![sr(0, 0, 10)],
                        }],
                    }),
                ],
            }
        }

        fn ctx_ranges() -> Vec<(u32, u64, u64)> {
            vec![(0, 0, 10), (1, 10, 20)]
        }

        /// Every categorical column in `index`, marked complete — what
        /// `build_category_dicts` produces for a file whose every shard carries
        /// the column's `CategoryBitset`. The default for these tests, which
        /// are about the row-set algebra rather than about trusting the index.
        fn complete_dicts(index: &PredicateIndex) -> CategoryDictionaries {
            let mut dicts = CategoryDictionaries::new();
            for col in &index.columns {
                if let crate::index::IndexedColumn::Categorical(cat) = col {
                    dicts.insert(
                        scx_format_io::column_name_hash(&cat.column_name),
                        cat.entries.iter().map(|e| e.value.clone()).collect(),
                        true,
                    );
                }
            }
            dicts
        }

        fn mk<'a>(
            index: &'a PredicateIndex,
            ranges: &'a [(u32, u64, u64)],
            dicts: &'a CategoryDictionaries,
        ) -> RowSetCtx<'a> {
            RowSetCtx {
                index,
                shard_row_ranges: ranges,
                n_obs: 20,
                category_dicts: dicts,
            }
        }

        fn eq(col: &str, v: &str) -> Predicate {
            Predicate::Eq(col.into(), ScalarValue::Utf8(v.into()))
        }

        /// An incomplete vocabulary is unusable here, and for a sharper reason
        /// than at Level-1.
        ///
        /// `categorical_eq` answers `Some(&[])` for an absent value, which this
        /// evaluator reads as an *exact* empty row-set — a wrong answer, not a
        /// slow one, when the vocabulary is only part of the column's values.
        /// The **hit** is no safer: a partial vocabulary's shard ranges cover
        /// only the rows that were recorded, so a present value resolves to a
        /// row-set missing the rest. Both must go residual.
        #[test]
        fn categorical_predicates_are_residual_when_the_vocabulary_is_incomplete() {
            let index = idx();
            let ranges = ctx_ranges();
            let mut dicts = complete_dicts(&index);
            // Mark `cell_type` incomplete; leave `tissue` alone as a control.
            dicts.insert(
                scx_format_io::column_name_hash("cell_type"),
                vec!["B cell".to_string(), "T cell".to_string()],
                false,
            );
            let ctx = mk(&index, &ranges, &dicts);

            assert!(
                eval_rowset(&eq("cell_type", "B cell"), &ctx).is_none(),
                "a value present in a partial vocabulary still resolves to only \
                 the rows that were recorded"
            );
            assert!(
                eval_rowset(&eq("cell_type", "Nope"), &ctx).is_none(),
                "absent from a partial vocabulary is not an exact empty row-set"
            );
            assert!(
                eval_rowset(
                    &Predicate::In("cell_type".into(), vec![ScalarValue::Utf8("B cell".into())]),
                    &ctx
                )
                .is_none(),
                "the `In` arm resolves through the same lookup"
            );
            assert!(
                eval_rowset(&eq("tissue", "blood"), &ctx).is_some(),
                "a complete sibling column must keep resolving — the guard is \
                 per column, not per file"
            );
        }

        /// A non-string `In` member cannot be *resolved* against a
        /// string-valued categorical index — it is unknown, not
        /// provably absent. Skipping it and returning the narrowed set
        /// silently drops rows; the whole predicate must go residual.
        ///
        /// Reachable since integer literals started validating against
        /// integer-valued categoricals: a file written
        /// before that fix carries an entry-less `CategoricalIndex` for
        /// such a column, so `batch in [1, 2]` would resolve to an
        /// exact **empty** row set.
        #[test]
        fn in_with_a_non_string_member_is_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let pred = Predicate::In(
                "cell_type".into(),
                vec![ScalarValue::Utf8("B cell".into()), ScalarValue::Int64(1)],
            );
            assert!(
                eval_rowset(&pred, &ctx).is_none(),
                "an unresolvable member must make the predicate residual, \
                 not narrow it to the members that happened to resolve"
            );
        }

        /// The same hazard in its purest form: an index whose column
        /// carries no entries at all (what earlier versions wrote for every
        /// integer categorical) must not answer "no rows anywhere".
        #[test]
        fn in_over_an_entry_less_categorical_is_residual() {
            let index = PredicateIndex {
                version: 1,
                columns: vec![IndexedColumn::Categorical(CategoricalIndex {
                    column_name: "batch".into(),
                    entries: vec![],
                })],
            };
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let pred = Predicate::In("batch".into(), vec![ScalarValue::Int64(1)]);
            assert!(
                eval_rowset(&pred, &ctx).is_none(),
                "an entry-less legacy index must fall back to a scan, not \
                 return an exact empty row set"
            );
            // The `In` arm was guarded by rejecting the non-string member. A
            // *string* literal walks straight past that guard and into
            // `categorical_eq`, whose miss on an empty entry list is reported
            // as an exact empty row-set — the same silent zero rows, one
            // spelling over.
            assert!(
                eval_rowset(&eq("batch", "1"), &ctx).is_none(),
                "a string literal reaches the same entry-less index through \
                 `Eq`, and must be residual there too"
            );
            assert!(
                eval_rowset(
                    &Predicate::In("batch".into(), vec![ScalarValue::Utf8("1".into())]),
                    &ctx
                )
                .is_none(),
                "and through `In` with a string member"
            );
        }

        #[test]
        fn eq_categorical_resolves_to_global_rowset() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let rs = eval_rowset(&eq("cell_type", "B cell"), &ctx).unwrap();
            // shard0 [0,5) + shard1 global [10,13)
            assert_eq!(
                rs.ranges(),
                &[
                    RowRange { start: 0, end: 5 },
                    RowRange { start: 10, end: 13 }
                ]
            );
        }

        #[test]
        fn eq_absent_value_is_exact_empty() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let rs = eval_rowset(&eq("cell_type", "NK cell"), &ctx).unwrap();
            assert!(rs.is_empty());
        }

        #[test]
        fn non_indexed_column_is_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            assert!(eval_rowset(&eq("donor_id", "d1"), &ctx).is_none());
        }

        #[test]
        fn in_list_unions() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let p = Predicate::In(
                "cell_type".into(),
                vec![
                    ScalarValue::Utf8("B cell".into()),
                    ScalarValue::Utf8("T cell".into()),
                ],
            );
            let rs = eval_rowset(&p, &ctx).unwrap();
            // B cell [0,5)+[10,13) ∪ T cell [5,10) = [0,13)
            assert_eq!(rs.ranges(), &[RowRange { start: 0, end: 13 }]);
        }

        #[test]
        fn and_intersects_or_unions() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            // cell_type==B cell AND tissue==blood -> [0,5) (shard1 not in blood)
            let and = Predicate::And(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("tissue", "blood")),
            );
            assert_eq!(
                eval_rowset(&and, &ctx).unwrap().ranges(),
                &[RowRange { start: 0, end: 5 }]
            );

            // cell_type==T cell OR tissue==blood -> [0,10)
            let or = Predicate::Or(
                Box::new(eq("cell_type", "T cell")),
                Box::new(eq("tissue", "blood")),
            );
            assert_eq!(
                eval_rowset(&or, &ctx).unwrap().ranges(),
                &[RowRange { start: 0, end: 10 }]
            );
        }

        #[test]
        fn or_resolves_from_index_regardless_of_column_nullability() {
            // `Or` used to be refused whenever any operand column was nullable
            // — not because the union was wrong, but so the fast path would
            // reproduce the mask path's non-Kleene `null OR true = false` bug.
            // With `or_kleene` the union IS the mask, so the refusal is gone
            // and a nullable categorical (the common case on an atlas: a
            // `cell_type` with unannotated cells) gets pushdown.
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let or = Predicate::Or(
                Box::new(eq("cell_type", "T cell")),
                Box::new(eq("tissue", "blood")),
            );
            assert_eq!(
                eval_rowset(&or, &ctx).unwrap().ranges(),
                &[RowRange { start: 0, end: 10 }],
                "an Or over a nullable categorical must resolve from the index"
            );
        }

        /// The `Or` union stops exactly at the Kleene TRUE-set, including at a
        /// null row — the property the whole row-set path rests on.
        ///
        /// `or_resolves_from_index_regardless_of_column_nullability` above only
        /// asserts that a nullable operand does not *forfeit* pushdown. It does
        /// not pin the boundary, and the boundary is where a wrong answer would
        /// live: the categorical index omits NULL rows, so a row that is
        /// UNKNOWN on one side and FALSE on the other must fall out of the
        /// union, not into it.
        ///
        /// Global rows 13..20 are exactly that row: no `cell_type` entry covers
        /// them (`B cell` ends at 13, `T cell` at 10) and `tissue == 'blood'` is
        /// shard 0 only. Rows 10..13 are the mirror case and must be kept —
        /// `B cell` is TRUE there even though `blood` is not.
        ///
        /// Today this is exercised only by the randomized generator in
        /// `tests/rowset_differential.rs`, where a seed or grammar change could
        /// lose it silently.
        #[test]
        fn or_stops_at_the_kleene_true_set_across_a_null_row() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let or = Predicate::Or(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("tissue", "blood")),
            );
            let rs = eval_rowset(&or, &ctx).expect("both sides resolve exactly");
            assert_eq!(
                rs.ranges(),
                &[RowRange { start: 0, end: 13 }],
                "union must be `B cell` [0,5)+[10,13) ∪ `blood` [0,10)"
            );
            assert!(
                rs.iter_rows().all(|r| r < 13),
                "a row that is UNKNOWN on one operand and FALSE on the other is \
                 not a match; rows 13..20 have no cell_type entry and no blood \
                 entry, so they must not be in the union"
            );
            assert!(
                (10..13).all(|r| rs.iter_rows().any(|x| x == r)),
                "rows 10..13 are TRUE on `cell_type == 'B cell'` even though \
                 `tissue == 'blood'` does not reach shard 1 — Kleene OR keeps \
                 them"
            );
        }

        #[test]
        fn and_with_residual_side_is_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let and = Predicate::And(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("donor_id", "d1")), // not indexed
            );
            assert!(eval_rowset(&and, &ctx).is_none());
        }

        /// `Not` must be answered without visiting its operand.
        ///
        /// The effect being guarded is **work not done**, which has no
        /// functional observable — `eval_rowset` returns `None` for a negated
        /// conjunct either way. So this counts `leaf` calls directly, through a
        /// stand-in algebra that mirrors `RowSetAlgebra`'s `short_circuit`.
        /// `not_short_circuits_before_visiting_its_operand` below then pins that
        /// the real algebra declares it; together they cover what an end-to-end
        /// assertion cannot see.
        ///
        /// Found by codex and Cursor Agent on PR #442: the first version of
        /// `walk` descended into `Not`, resolved the inner leaf, allocated and
        /// sorted a `RowSet` from every matching `ShardRange`, discarded it, and
        /// then took the residual decode path anyway.
        #[test]
        fn walk_does_not_visit_operands_of_a_short_circuited_node() {
            use crate::rowset::RowSet;
            use std::cell::Cell;

            struct Counting {
                leaf_calls: Cell<usize>,
            }
            impl crate::predicate::PredicateAlgebra for Counting {
                type Value = Option<RowSet>;
                fn leaf(&self, _pred: &Predicate) -> crate::error::Result<Self::Value> {
                    self.leaf_calls.set(self.leaf_calls.get() + 1);
                    Ok(Some(RowSet::empty()))
                }
                fn and(
                    &self,
                    _a: Self::Value,
                    _b: Self::Value,
                ) -> crate::error::Result<Self::Value> {
                    Ok(Some(RowSet::empty()))
                }
                fn or(
                    &self,
                    _a: Self::Value,
                    _b: Self::Value,
                ) -> crate::error::Result<Self::Value> {
                    Ok(Some(RowSet::empty()))
                }
                fn not(&self, _a: Self::Value) -> crate::error::Result<Self::Value> {
                    Ok(None)
                }
                fn short_circuit(&self, pred: &Predicate) -> Option<Self::Value> {
                    match pred {
                        Predicate::Not(_) => Some(None),
                        _ => None,
                    }
                }
            }

            let inner = Predicate::And(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("tissue", "blood")),
            );

            // Control: the same subtree, un-negated, visits both leaves. Without
            // this the assertion below passes against a `walk` that visits
            // nothing at all.
            let alg = Counting {
                leaf_calls: Cell::new(0),
            };
            crate::predicate::walk(&inner, &alg).unwrap();
            assert_eq!(
                alg.leaf_calls.get(),
                2,
                "the un-negated subtree must visit both leaves"
            );

            let alg = Counting {
                leaf_calls: Cell::new(0),
            };
            let out = crate::predicate::walk(&Predicate::Not(Box::new(inner)), &alg).unwrap();
            assert_eq!(
                alg.leaf_calls.get(),
                0,
                "`Not` is residual whatever its operand resolves to, so `walk` \
                 must not descend into it — resolving the operand allocates and \
                 sorts a RowSet that is then discarded"
            );
            assert!(out.is_none(), "`Not` still resolves to residual");
        }

        /// The real `RowSetAlgebra` declares `Not` short-circuiting, and only
        /// `Not`. Pairs with the counting test above: that one proves `walk`
        /// honours the hook, this one proves the row-set algebra sets it.
        #[test]
        fn not_short_circuits_before_visiting_its_operand() {
            use crate::predicate::PredicateAlgebra;
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            let alg = crate::predicate::RowSetAlgebra { ctx: &ctx };

            assert!(
                matches!(
                    alg.short_circuit(&Predicate::Not(Box::new(eq("cell_type", "B cell")))),
                    Some(None)
                ),
                "Not must be answered without descending"
            );
            for p in [
                eq("cell_type", "B cell"),
                Predicate::Ne("cell_type".into(), ScalarValue::Utf8("B cell".into())),
                Predicate::And(
                    Box::new(eq("cell_type", "B cell")),
                    Box::new(eq("tissue", "blood")),
                ),
                Predicate::Or(
                    Box::new(eq("cell_type", "B cell")),
                    Box::new(eq("tissue", "blood")),
                ),
            ] {
                assert!(
                    alg.short_circuit(&p).is_none(),
                    "only Not short-circuits; {p} must still be walked"
                );
            }
        }

        #[test]
        fn ne_not_numeric_are_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let dicts = complete_dicts(&index);
            let ctx = mk(&index, &ranges, &dicts);
            assert!(eval_rowset(
                &Predicate::Ne("cell_type".into(), ScalarValue::Utf8("B cell".into())),
                &ctx
            )
            .is_none());
            assert!(
                eval_rowset(&Predicate::Not(Box::new(eq("cell_type", "B cell"))), &ctx).is_none()
            );
            assert!(eval_rowset(
                &Predicate::Gt("cell_type".into(), ScalarValue::Int64(5)),
                &ctx
            )
            .is_none());
        }
    }
}

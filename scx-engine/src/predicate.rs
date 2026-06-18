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

use arrow::array::{Array, AsArray, BooleanArray, RecordBatch};
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
// docs in `collect.rs` and the staleness/null reasoning in the plan: the
// categorical index omits null rows, so complementing it (for `Ne`/`Not`) would
// wrongly re-include nulls, and numeric B+ tree leaves are conservative.
// ---------------------------------------------------------------------------

use crate::index::PredicateIndex;
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
    /// The obs schema, used to check column nullability. The legacy mask
    /// evaluator uses Arrow's non-Kleene `or` (`null OR true = null → false`),
    /// so an `Or` over a nullable column would diverge from the row-set union
    /// (which would include such rows). `Or` is therefore index-resolvable only
    /// when every referenced column is non-nullable — see [`eval_rowset`].
    pub obs_schema: &'a Schema,
}

impl RowSetCtx<'_> {
    /// True if every column referenced by `pred` is non-nullable in the obs
    /// schema. A column missing from the schema is treated as nullable
    /// (conservative → residual).
    fn all_columns_non_nullable(&self, pred: &Predicate) -> bool {
        pred.columns().iter().all(|c| {
            self.obs_schema
                .field_with_name(c)
                .map(|f| !f.is_nullable())
                .unwrap_or(false)
        })
    }
}

impl RowSetCtx<'_> {
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

/// Resolve `pred` to an exact global [`RowSet`] using only the index, or `None`
/// if any node touches a non-indexed column or an op outside the v1 exact
/// scope (then the whole subtree is residual).
pub fn eval_rowset(pred: &Predicate, ctx: &RowSetCtx) -> Option<RowSet> {
    match pred {
        Predicate::Eq(col, ScalarValue::Utf8(v)) => {
            // `categorical_eq` returns None iff the column is not an indexed
            // categorical (residual); Some(&[]) iff indexed but value absent
            // (exact empty row-set).
            let ranges = ctx.index.categorical_eq(col, v)?;
            ctx.shard_ranges_to_rowset(ranges)
        }
        // Eq on a categorical column with a non-string literal cannot match a
        // string category; only string equality is index-resolvable here.
        Predicate::Eq(_, _) => None,
        Predicate::In(col, vals) => {
            // Only index-resolvable if the column is an indexed categorical.
            if ctx.index.indexed_kind(col) != Some(crate::index::IndexKind::Categorical) {
                return None;
            }
            let mut acc = RowSet::empty();
            for v in vals {
                if let ScalarValue::Utf8(s) = v {
                    let ranges = ctx.index.categorical_eq(col, s)?;
                    acc = acc.union(&ctx.shard_ranges_to_rowset(ranges)?);
                }
                // non-string members can't match a string categorical → skip
            }
            Some(acc)
        }
        Predicate::And(a, b) => {
            let ra = eval_rowset(a, ctx)?;
            let rb = eval_rowset(b, ctx)?;
            Some(ra.intersect(&rb))
        }
        Predicate::Or(a, b) => {
            // An Or with a residual side can match rows in ANY shard, so it is
            // not narrowable: both sides must resolve exactly.
            //
            // Null semantics: the legacy mask evaluator uses Arrow's non-Kleene
            // `or`, where `null OR x = null → false`. So a row whose left
            // operand is null but whose right operand is true is EXCLUDED by the
            // legacy path, whereas the row-set union would INCLUDE it. They
            // diverge only when an operand column is nullable, so resolve `Or`
            // from the index only when every referenced column is non-nullable
            // (then no operand can be null and union == legacy). Otherwise fall
            // back to the residual path, which preserves the exact legacy
            // semantics.
            if !ctx.all_columns_non_nullable(a) || !ctx.all_columns_non_nullable(b) {
                return None;
            }
            let ra = eval_rowset(a, ctx)?;
            let rb = eval_rowset(b, ctx)?;
            Some(ra.union(&rb))
        }
        // Residual in v1 (see module docs): Ne/Not (null-complement hazard),
        // numeric comparisons (conservative B+ tree leaves).
        Predicate::Ne(_, _)
        | Predicate::Not(_)
        | Predicate::Lt(_, _)
        | Predicate::Gt(_, _)
        | Predicate::Le(_, _)
        | Predicate::Ge(_, _) => None,
    }
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
    fn validate_type(
        &self,
        col: &str,
        col_type: &DataType,
        value: &ScalarValue,
    ) -> Result<ScalarValue> {
        match (col_type, value) {
            // String columns accept string values
            (DataType::Utf8 | DataType::LargeUtf8, ScalarValue::Utf8(_)) => Ok(value.clone()),
            // Dictionary columns: compare against dictionary values (typically strings)
            (DataType::Dictionary(_, value_type), ScalarValue::Utf8(_))
                if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
            {
                Ok(value.clone())
            }
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
/// Null values produce `false` (not matched). Nulls propagate through AND/OR/NOT
/// and are coalesced to false at the end.
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

/// Internal evaluator that preserves null identity through the expression tree.
fn eval_inner(predicate: &Predicate, batch: &RecordBatch) -> Result<BooleanArray> {
    match predicate {
        Predicate::Eq(col, val) => eval_comparison(batch, col, val, CmpOp::Eq),
        Predicate::Ne(col, val) => eval_comparison(batch, col, val, CmpOp::Ne),
        Predicate::Lt(col, val) => eval_comparison(batch, col, val, CmpOp::Lt),
        Predicate::Gt(col, val) => eval_comparison(batch, col, val, CmpOp::Gt),
        Predicate::Le(col, val) => eval_comparison(batch, col, val, CmpOp::Le),
        Predicate::Ge(col, val) => eval_comparison(batch, col, val, CmpOp::Ge),
        Predicate::In(col, vals) => {
            // OR of individual Eq comparisons
            let mut result: Option<BooleanArray> = None;
            for val in vals {
                let mask = eval_comparison(batch, col, val, CmpOp::Eq)?;
                result = Some(match result {
                    None => mask,
                    Some(prev) => compute::kernels::boolean::or(&prev, &mask)?,
                });
            }
            // If vals is empty, return all-false
            Ok(result.unwrap_or_else(|| BooleanArray::from(vec![false; batch.num_rows()])))
        }
        Predicate::And(a, b) => {
            let left = eval_inner(a, batch)?;
            let right = eval_inner(b, batch)?;
            Ok(compute::kernels::boolean::and(&left, &right)?)
        }
        Predicate::Or(a, b) => {
            let left = eval_inner(a, batch)?;
            let right = eval_inner(b, batch)?;
            Ok(compute::kernels::boolean::or(&left, &right)?)
        }
        Predicate::Not(inner) => {
            let mask = eval_inner(inner, batch)?;
            // Arrow's `not` preserves nulls: null stays null, true→false, false→true
            Ok(compute::kernels::boolean::not(&mask)?)
        }
    }
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
    let column = batch.column(col_idx);
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
            DataType::Utf8 | DataType::LargeUtf8 => eval_dictionary_utf8(column, value, op),
            _ => Err(EngineError::SchemaError {
                column: col_name.to_string(),
                reason: format!("unsupported dictionary value type: {value_type:?}"),
            }),
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
    use arrow::array::{DictionaryArray, Int32Array, Int64Array, StringArray};
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

        /// Schema with cell_type/tissue/donor_id all NON-nullable so `Or` is
        /// index-resolvable (the nullable-Or case is exercised separately).
        fn test_schema() -> arrow::datatypes::Schema {
            use arrow::datatypes::{DataType, Field};
            arrow::datatypes::Schema::new(vec![
                Field::new("cell_type", DataType::Utf8, false),
                Field::new("tissue", DataType::Utf8, false),
                Field::new("donor_id", DataType::Utf8, false),
            ])
        }

        fn mk<'a>(
            index: &'a PredicateIndex,
            ranges: &'a [(u32, u64, u64)],
            schema: &'a arrow::datatypes::Schema,
        ) -> RowSetCtx<'a> {
            RowSetCtx {
                index,
                shard_row_ranges: ranges,
                n_obs: 20,
                obs_schema: schema,
            }
        }

        fn eq(col: &str, v: &str) -> Predicate {
            Predicate::Eq(col.into(), ScalarValue::Utf8(v.into()))
        }

        #[test]
        fn eq_categorical_resolves_to_global_rowset() {
            let index = idx();
            let ranges = ctx_ranges();
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
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
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
            let rs = eval_rowset(&eq("cell_type", "NK cell"), &ctx).unwrap();
            assert!(rs.is_empty());
        }

        #[test]
        fn non_indexed_column_is_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
            assert!(eval_rowset(&eq("donor_id", "d1"), &ctx).is_none());
        }

        #[test]
        fn in_list_unions() {
            let index = idx();
            let ranges = ctx_ranges();
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
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
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
            // cell_type==B cell AND tissue==blood -> [0,5) (shard1 not in blood)
            let and = Predicate::And(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("tissue", "blood")),
            );
            assert_eq!(
                eval_rowset(&and, &ctx).unwrap().ranges(),
                &[RowRange { start: 0, end: 5 }]
            );

            // cell_type==T cell OR tissue==blood -> [0,10) (both columns non-nullable)
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
        fn or_over_nullable_column_is_residual() {
            // When an Or operand references a NULLABLE column, the legacy
            // non-Kleene `or` (null OR true = false) diverges from the row-set
            // union, so the Or must fall back to residual.
            use arrow::datatypes::{DataType, Field};
            let index = idx();
            let ranges = ctx_ranges();
            let schema = arrow::datatypes::Schema::new(vec![
                Field::new("cell_type", DataType::Utf8, true), // nullable
                Field::new("tissue", DataType::Utf8, false),
            ]);
            let ctx = mk(&index, &ranges, &schema);
            let or = Predicate::Or(
                Box::new(eq("cell_type", "T cell")),
                Box::new(eq("tissue", "blood")),
            );
            assert!(eval_rowset(&or, &ctx).is_none());
        }

        #[test]
        fn and_with_residual_side_is_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
            let and = Predicate::And(
                Box::new(eq("cell_type", "B cell")),
                Box::new(eq("donor_id", "d1")), // not indexed
            );
            assert!(eval_rowset(&and, &ctx).is_none());
        }

        #[test]
        fn ne_not_numeric_are_residual() {
            let index = idx();
            let ranges = ctx_ranges();
            let schema = test_schema();
            let ctx = mk(&index, &ranges, &schema);
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

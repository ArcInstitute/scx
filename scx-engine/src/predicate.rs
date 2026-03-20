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

        return Err(EngineError::PredicateParseError {
            expr: expr.to_string(),
            reason: format!("unexpected character: '{}'", chars[i]),
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
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, schema: &'a Schema, expr: &str) -> Self {
        Self {
            tokens,
            pos: 0,
            schema,
            expr: expr.to_string(),
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
            None => Err(EngineError::SchemaError {
                column: col.to_string(),
                reason: "column not found in schema".to_string(),
            }),
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
pub fn parse_predicate(expr: &str, schema: &Schema) -> Result<Predicate> {
    if expr.trim().is_empty() {
        return Err(EngineError::PredicateParseError {
            expr: expr.to_string(),
            reason: "empty predicate expression".to_string(),
        });
    }

    let tokens = tokenize(expr)?;
    let mut parser = Parser::new(tokens, schema, expr);
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
                // Use as_usize for ArrowNativeType, then cast to f64.
                // This handles all numeric types: i8-i64, u8-u64, f32, f64.
                let native = array.value(i);
                let v = native_to_f64(native);
                Some(match op {
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

/// Convert any ArrowNativeType to f64 using byte reinterpretation.
/// This handles all integer types (i8–i64, u8–u64) and float types (f32, f64).
fn native_to_f64<N: arrow::datatypes::ArrowNativeType>(v: N) -> f64 {
    // ArrowNativeType doesn't directly provide a to_f64 method.
    // Use the fact that all Arrow integer types implement Into<i128> or similar.
    // The cleanest approach: use the Debug trait to convert via string.
    // Actually, use std::mem::size_of + byte casting for performance.
    // But the simplest and most correct approach: each ArrowPrimitiveType's
    // Native has a known set of possible types. We handle them at the call
    // site via monomorphization — each concrete type will compile to a
    // direct cast. We use `as` for this at the eval_numeric_array dispatch
    // site instead.
    //
    // Fallback: since ArrowNativeType: Copy + Debug, and we know the
    // concrete types are all numeric primitives, we parse from Debug output.
    // This is only called for value comparisons, not hot-path data.
    use std::any::Any;
    let any_ref: &dyn Any = &v;
    if let Some(&val) = any_ref.downcast_ref::<i8>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<i16>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<i32>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<i64>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<u8>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<u16>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<u32>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<u64>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<f32>() {
        return val as f64;
    }
    if let Some(&val) = any_ref.downcast_ref::<f64>() {
        return val;
    }
    // half::f16 is also possible but unlikely in predicate evaluation
    0.0 // unreachable for supported types
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
        let pred = parse_predicate("cell_type == 'T cell'", &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()))
        );
    }

    #[test]
    fn parse_double_quoted_string() {
        let schema = test_schema();
        let pred = parse_predicate("cell_type == \"T cell\"", &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::Eq("cell_type".into(), ScalarValue::Utf8("T cell".into()))
        );
    }

    #[test]
    fn parse_and_comparison() {
        let schema = test_schema();
        let pred = parse_predicate("n_genes > 200 and n_counts < 5000", &schema).unwrap();
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
        let pred =
            parse_predicate("cell_type in ['T cell', 'B cell', 'NK cell']", &schema).unwrap();
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
        let pred = parse_predicate("not is_doublet == true", &schema).unwrap();
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
        let err = parse_predicate("nonexistent_column == 'x'", &schema).unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
    }

    #[test]
    fn parse_type_mismatch_string_vs_int() {
        let schema = test_schema();
        let err = parse_predicate("cell_type > 5", &schema).unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
    }

    #[test]
    fn parse_empty_string_error() {
        let schema = test_schema();
        let err = parse_predicate("", &schema).unwrap_err();
        assert!(matches!(err, EngineError::PredicateParseError { .. }));
    }

    #[test]
    fn parse_malformed_error() {
        let schema = test_schema();
        let err = parse_predicate("cell_type ==", &schema).unwrap_err();
        assert!(matches!(err, EngineError::PredicateParseError { .. }));
    }

    #[test]
    fn parse_float_value() {
        let schema = test_schema();
        let pred = parse_predicate("score > 0.5", &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::Gt("score".into(), ScalarValue::Float64(0.5))
        );
    }

    #[test]
    fn parse_le_ge() {
        let schema = test_schema();
        let pred = parse_predicate("n_genes >= 100 and n_genes <= 5000", &schema).unwrap();
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
}

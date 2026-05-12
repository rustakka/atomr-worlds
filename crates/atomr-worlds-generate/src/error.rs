#[derive(Debug, thiserror::Error)]
pub enum GenerateError {
    #[error("invalid configuration: {0}")]
    BadConfig(&'static str),
}

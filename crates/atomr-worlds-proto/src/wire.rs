//! Encode/decode helpers wrapping `bincode` 2's serde feature.

use serde::{de::DeserializeOwned, Serialize};

use crate::error::ProtoError;

#[inline]
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, ProtoError> {
    let cfg = bincode::config::standard();
    Ok(bincode::serde::encode_to_vec(value, cfg)?)
}

#[inline]
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ProtoError> {
    let cfg = bincode::config::standard();
    let (val, _read) = bincode::serde::decode_from_slice::<T, _>(bytes, cfg)?;
    Ok(val)
}

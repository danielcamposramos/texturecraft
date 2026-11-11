use super::{MAPS_LIMIT_MSG, MAX_MAPS, MapEntry};

pub(super) struct MapEntriesVisitor;

impl<'de> serde::de::Visitor<'de> for MapEntriesVisitor {
    type Value = Vec<MapEntry>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a list of maps")
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error> {
        let mut maps = Vec::new();
        while let Some(entry) = sequence.next_element()? {
            if maps.len() == MAX_MAPS {
                return Err(serde::de::Error::custom(MAPS_LIMIT_MSG));
            }

            maps.push(entry);
        }

        Ok(maps)
    }
}

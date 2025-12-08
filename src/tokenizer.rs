use std::{collections::HashMap};
use bytes::Bytes;


trait Tokenize {
    fn train(&mut self, text: String, vocab_size: usize);
    fn encode(&self, text: String) -> Vec<u32>;
    fn decode(&self, ids: &Vec<u32>) -> String;
}

struct BasicTokenizer {
    vocab: HashMap<u32, Bytes>,
    merges: HashMap<(u32, u32), u32>
}

impl BasicTokenizer {
    fn new() -> Self {
        BasicTokenizer { vocab: HashMap::new(), merges: HashMap::new() }
    }
}

fn merge(ids: &[u32], pair: (u32, u32), idx: u32) -> Vec<u32> {
    let mut new_ids: Vec<u32> = Vec::with_capacity(ids.len());
    
    let mut merged = false;
    // Sliding window of size 2 to get all bigrams
    for window in ids.windows(2) {
        if merged { 
            merged = false; 
            continue
        }

        let (b1, b2) = (window[0], window[1]);
        if pair.0 == b1 && pair.1 == b2 {
            new_ids.push(idx);
            merged = true
        }
        else {
            new_ids.push(b1);
        }
    }

    new_ids.push(ids[ids.len() - 1]);
    new_ids
}

fn get_stats(ids: &[u32], stats: &mut HashMap<(u32, u32), u32>) {
    for window in ids.windows(2) {
        *stats.entry((window[0], window[1])).or_insert(0) += 1;
    }
}

impl Tokenize for BasicTokenizer {
    fn train(&mut self, text: String, vocab_size: usize) {
        let num_merges: usize = vocab_size - 256;
        let ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        
        // Initialize vocab with single value bytes
        for i in 0..256 {
            self.vocab.insert(i, Bytes::copy_from_slice(&[i as u8]));
        }

        let mut stats = HashMap::new();
        get_stats(&ids, &mut stats);

    }

    fn encode(&self, text: String) -> Vec<u32> {
        todo!()
    }

    fn decode(&self, ids: &Vec<u32>) -> String {
        todo!()
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_merge() {
        let merged = merge(&[1, 2, 3, 4, 4, 4, 5, 6], (3, 4), 10);
        assert_eq!(merged, [1, 2, 10, 4, 4, 5, 6]);
    }
}

use std::{collections::HashMap};
use bytes::{Bytes, BytesMut};
use regex::Regex;
use std::io::{BufWriter, Write};


pub trait Tokenize {
    fn train(&mut self, text: &str, vocab_size: usize, verbose: bool);
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn save(&self, vocab_path: &str, merges_path: &str) -> std::io::Result<()>;
}

pub struct BasicTokenizer {
    vocab: HashMap<u32, Bytes>,
    merges: HashMap<(u32, u32), u32>
}

impl BasicTokenizer {
    pub fn new() -> Self {
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
    fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        let num_merges: usize = vocab_size - 256;
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        
        // Initialize vocab with single value bytes
        for i in 0..256 {
            self.vocab.insert(i, Bytes::copy_from_slice(&[i as u8]));
        }


        for i in 0..num_merges {
            // Update the bigrams counts hashmap
            let mut stats = HashMap::new();
            get_stats(&ids, &mut stats);
            
            // Take the bigram that occuress more oftent
            let pair = stats.iter().max_by_key(|(_, v)| *v).map(|(k, _)| *k).unwrap();
            
            let minted_idx = self.vocab.len() as u32;
            ids = merge(&ids, pair, minted_idx);

            // Concatenate bytes pair
            let (b1, b2) = (self.vocab.get(&pair.0).unwrap(), self.vocab.get(&pair.1).unwrap());
            let mut buf: BytesMut = bytes::BytesMut::with_capacity(b1.len() + b2.len());
            buf.extend_from_slice(b1.as_ref());
            buf.extend_from_slice(b2.as_ref());
            let concat_bytes: Bytes = buf.freeze();

            self.vocab.insert(minted_idx, concat_bytes);
            self.merges.insert(pair, minted_idx);

            if verbose {
                let minted_token = str::from_utf8(self.vocab.get(&minted_idx).unwrap().as_ref()).unwrap();
                let percentage = (i + 1) as f64 / num_merges as f64 * 100.0;
                println!("{}/{} - {:.2}%", i+1, num_merges, percentage);
                println!("Merged [{}] [{}] -> [{}] ({})",  pair.0, pair.1, minted_idx, minted_token);
            }
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Converting the text bytes to integers
        let mut ids: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
        while ids.len() > 1 {
            let bigrams: Vec<&[u32]> = ids.windows(2).collect();
            
            // Get the merged pair with the lowest idx (which is a reference to a reference to a slice)
            let pair = bigrams.iter().min_by_key(
                |&&bigram| self.merges.get(&(bigram[0], bigram[1])).unwrap_or(&u32::MAX)
            );

            match pair {
                Some(&pair) => {
                    let (idx1, idx2) = (pair[0], pair[1]);
                    if !self.merges.contains_key(&(idx1, idx2)) { break }
                    ids = merge(&ids, (idx1, idx2), self.merges[&(idx1, idx2)]);
                }
                None => { break }
            }
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut buf = BytesMut::new();
        for idx in ids {
            buf.extend_from_slice(self.vocab[idx].as_ref());
        }
        String::from_utf8(buf.freeze().to_vec()).unwrap()
    }

    fn save(&self, vocab_path: &str, merges_path: &str) -> std::io::Result<()> {
        let file = std::fs::File::create(vocab_path)?;
        let mut writer = BufWriter::new(file);

        for (&idx, bytes) in self.vocab.iter() {
            let content: std::borrow::Cow<str> = String::from_utf8_lossy(bytes);
            writer.write_all(format!("[{}] -> ({})\n", idx, content).as_bytes())?;
        }
        writer.flush()?;

        let file = std::fs::File::create(merges_path)?;
        writer = BufWriter::new(file);

        for (&(idx1, idx2), &idx_minted) in self.merges.iter() {
            writer.write_all(format!("[{idx1}][{idx2}] -> [{idx_minted}]\n").as_bytes())?;
        }
        writer.flush()?;

        Ok(())
    }
}


const GPT4_SPLIT_PATTERN: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
);

struct RegexTokenizer {
    inner: BasicTokenizer,
    regex: Regex
}

impl RegexTokenizer {
    fn new(pattern: &str) -> Result<Self, regex::Error> {
        Ok(RegexTokenizer { inner: BasicTokenizer::new(), regex: Regex::new(pattern)? })
    }
}

impl Tokenize for RegexTokenizer {
    fn train(&mut self, text: &str, vocab_size: usize, verbose: bool) {
        todo!()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        todo!()
    }

    fn decode(&self, ids: &[u32]) -> String {
        todo!()
    }

    fn save(&self, vocab_path: &str, merges_path: &str) -> std::io::Result<()> {
        self.inner.save(vocab_path, merges_path)
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

    #[test]
    fn test_train() {
        let text = std::fs::read_to_string("train.txt").expect("Failed to read file");
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.train(&text, 500, false);
    }

    #[test] 
    fn test_encode() {
        let text = std::fs::read_to_string("train.txt").expect("Failed to read file");
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.train(&text, 500, false);
        println!("{:?}", tokenizer.encode("Self driving is the future! 🙄"));
    }

    #[test] 
    fn test_decode() {
        let text = std::fs::read_to_string("train.txt").expect("Failed to read file");
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.train(&text, 500, false);
        let s = "Self driving is the future! 🙄";

        let encoded = tokenizer.encode(s);
        assert_eq!(s, tokenizer.decode(&encoded));
    }
}

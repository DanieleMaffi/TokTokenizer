use toktokenizer::tokenizer::{BasicTokenizer, Tokenize};

fn main() {
    let mut tknzr = BasicTokenizer::new();
    let text = std::fs::read_to_string("train.txt").expect("Failed to read file");
    tknzr.train(&text, 500, true);
    tknzr.save("vocab.model", "merges.txt").expect("Could not save tokenizer");
}


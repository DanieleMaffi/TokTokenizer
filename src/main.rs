mod tokenizer;


fn main() {
    let a = b"ciao";
    println!("{a:?}");
    for b in a {
        print!("{b:08b} ");
    }
}


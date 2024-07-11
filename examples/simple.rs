// extern crate math_accelerate;

// use math_accelerate::MathAccelerator;

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> AnyResult<()> {
    print_values(&[&7i16, &8u16]);
    Ok(())
}

fn print_values(slice: &[&dyn ToSomething]) {
    for &a in slice {
        let s: Something = a.to_something();
        if let Some(value) = s.i16 {
            println!("i16: {}", value);
        }
        if let Some(value) = s.u16 {
            println!("u16: {}", value);
        }
    }
}

trait ToSomething {
    fn to_something(&self) -> Something;
}

struct Something {
    i16: Option<i16>,
    u16: Option<u16>,
}

impl ToSomething for i16 {
    fn to_something(&self) -> Something {
        Something {
            i16: Some(*self),
            u16: None,
        }
    }
}

impl ToSomething for u16 {
    fn to_something(&self) -> Something {
        Something {
            i16: None,
            u16: Some(*self),
        }
    }
}

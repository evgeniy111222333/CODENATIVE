def repeat_values() -> tuple[str, int]:
    shared_name = "echo"
    first = shared_name
    second = shared_name
    repeated = "echo"
    count = 42
    mirror = 42
    return first + second + repeated, count + mirror

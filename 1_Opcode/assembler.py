import sys
from typing import List, Dict, Tuple

# LC-3b assembler implemented in Python.
# The assembler performs two passes: first to build the symbol table
# and second to emit machine code into the requested output file.

OPCODES = {
    "add",
    "and",
    "br",
    "brn",
    "brz",
    "brp",
    "brnz",
    "brnp",
    "brzp",
    "brnzp",
    "halt",
    "jmp",
    "jsr",
    "jsrr",
    "ldb",
    "ldw",
    "lea",
    "nop",
    "not",
    "ret",
    "lshf",
    "rshfl",
    "rshfa",
    "rti",
    "stb",
    "stw",
    "trap",
    "xor",
}

PSEUDO_OPS = {".orig", ".end", ".fill"}

REGISTER_PREFIX = "r"


class AssemblyError(Exception):
    """Custom exception used for assembler errors."""


# --------------------------- Utility helpers ---------------------------

def tokenize(line: str) -> List[str]:
    """Split a line into tokens, removing comments and commas."""
    code = line.split(";", 1)[0]
    code = code.replace(",", " ")
    return [tok for tok in code.strip().split() if tok]


def is_register(token: str) -> bool:
    return token.lower().startswith(REGISTER_PREFIX) and token[1:].isdigit() and 0 <= int(token[1:]) <= 7


def parse_register(token: str) -> int:
    if not is_register(token):
        raise AssemblyError(f"Invalid register {token}")
    return int(token[1:])


def parse_constant(token: str) -> int:
    """Parse a numeric constant of the form #number or xHEX, with optional sign."""
    if token.startswith("#"):
        return int(token[1:], 10)
    if token.startswith("x"):
        # Support x-FF syntax where the sign follows the x prefix.
        sign = 1
        digits = token[1:]
        if digits.startswith("-"):
            sign = -1
            digits = digits[1:]
        return sign * int(digits, 16)
    raise AssemblyError(f"Invalid constant {token}")


def sign_extend(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= (1 << bits)
    return value


# --------------------------- Symbol table pass ---------------------------

def build_symbol_table(lines: List[str]) -> Tuple[Dict[str, int], int]:
    """
    First pass: build a symbol table mapping labels to addresses.
    Returns the table and the starting address from .ORIG.
    """
    symbols: Dict[str, int] = {}
    address = None
    started = False

    for raw in lines:
        tokens = tokenize(raw.lower())
        if not tokens:
            continue
        if not started:
            if tokens[0] == ".orig":
                if len(tokens) < 2:
                    raise AssemblyError(".orig requires an address")
                address = parse_constant(tokens[1])
                started = True
            continue

        # Stop processing after .end
        if tokens[0] == ".end":
            break

        label = None
        opcode_index = 0
        if tokens[0] not in OPCODES and tokens[0] not in PSEUDO_OPS:
            label = tokens[0]
            opcode_index = 1

        if label:
            if label in symbols:
                raise AssemblyError(f"Duplicate label {label}")
            symbols[label] = address

        if opcode_index >= len(tokens):
            continue

        opcode = tokens[opcode_index]
        if opcode == ".fill" or opcode in OPCODES:
            address += 2
        # Other pseudo-ops (.end) handled above

    if address is None:
        raise AssemblyError("No .orig found in source")
    return symbols, address


# --------------------------- Encoding helpers ---------------------------

def pc_offset(target: int, current_address: int, bits: int) -> int:
    offset = (target - (current_address + 2)) // 2
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    if offset < min_val or offset > max_val:
        raise AssemblyError(f"Offset {offset} out of range for {bits} bits")
    return offset & ((1 << bits) - 1)


def encode_trap(vector_token: str) -> int:
    trap_vector = parse_constant(vector_token)
    if trap_vector < 0 or trap_vector > 0xFF:
        raise AssemblyError("Trap vector out of range")
    return 0xF000 | trap_vector


def encode_shift(opcode: int, dr: int, sr: int, mode_bits: int, amount: int) -> int:
    if not (0 <= amount <= 15):
        raise AssemblyError("Shift amount must be 0-15")
    return (opcode << 12) | (dr << 9) | (sr << 6) | (mode_bits << 4) | amount


# --------------------------- Second pass ---------------------------

def assemble(lines: List[str], symbols: Dict[str, int], output_path: str) -> None:
    started = False
    current_address = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for raw in lines:
            tokens = tokenize(raw.lower())
            if not tokens:
                continue
            if not started:
                if tokens[0] == ".orig":
                    current_address = parse_constant(tokens[1])
                    out.write(f"0x{current_address:04X}\n")
                    started = True
                continue

            if tokens[0] == ".end":
                break

            label = None
            opcode_index = 0
            if tokens[0] not in OPCODES and tokens[0] not in PSEUDO_OPS:
                label = tokens[0]
                opcode_index = 1

            if opcode_index >= len(tokens):
                continue

            opcode = tokens[opcode_index]
            operands = tokens[opcode_index + 1 :]

            machine_word = None

            if opcode == ".fill":
                value_token = operands[0]
                if value_token in symbols:
                    value = symbols[value_token]
                else:
                    value = parse_constant(value_token)
                machine_word = value & 0xFFFF

            elif opcode.startswith("br"):
                cond_str = opcode[2:]
                n = 1 if ("n" in cond_str or cond_str == "") else 0
                z = 1 if ("z" in cond_str or cond_str == "") else 0
                p = 1 if ("p" in cond_str or cond_str == "") else 0
                if cond_str == "":
                    n = z = p = 1
                target = symbols[operands[0]] if operands[0] in symbols else parse_constant(operands[0])
                offset = pc_offset(target, current_address, 9)
                machine_word = (0 << 12) | (n << 11) | (z << 10) | (p << 9) | offset

            elif opcode == "add" or opcode == "and" or opcode == "xor":
                dr = parse_register(operands[0])
                sr1 = parse_register(operands[1])
                if is_register(operands[2]):
                    sr2 = parse_register(operands[2])
                    machine_word = (
                        (1 if opcode == "add" else 5 if opcode == "and" else 9) << 12
                    ) | (dr << 9) | (sr1 << 6) | sr2
                else:
                    imm5 = parse_constant(operands[2])
                    imm5 = sign_extend(imm5, 5) & 0x1F
                    machine_word = (
                        (1 if opcode == "add" else 5 if opcode == "and" else 9) << 12
                    ) | (dr << 9) | (sr1 << 6) | 0x20 | imm5

            elif opcode == "not":
                dr = parse_register(operands[0])
                sr = parse_register(operands[1])
                machine_word = (9 << 12) | (dr << 9) | (sr << 6) | 0x3F

            elif opcode == "jmp":
                base = parse_register(operands[0])
                machine_word = (12 << 12) | (base << 6)

            elif opcode == "ret":
                machine_word = (12 << 12) | (7 << 6)

            elif opcode == "jsr":
                target = symbols[operands[0]] if operands[0] in symbols else parse_constant(operands[0])
                offset = pc_offset(target, current_address, 11)
                machine_word = (4 << 12) | (1 << 11) | offset

            elif opcode == "jsrr":
                base = parse_register(operands[0])
                machine_word = (4 << 12) | (base << 6)

            elif opcode == "lea":
                dr = parse_register(operands[0])
                target = symbols[operands[1]] if operands[1] in symbols else parse_constant(operands[1])
                offset = pc_offset(target, current_address, 9)
                machine_word = (14 << 12) | (dr << 9) | offset

            elif opcode == "ldw" or opcode == "stw":
                reg = parse_register(operands[0])
                base = parse_register(operands[1])
                offset6 = sign_extend(parse_constant(operands[2]), 6) & 0x3F
                op_val = 6 if opcode == "ldw" else 7
                machine_word = (op_val << 12) | (reg << 9) | (base << 6) | offset6

            elif opcode == "ldb" or opcode == "stb":
                reg = parse_register(operands[0])
                base = parse_register(operands[1])
                offset6 = sign_extend(parse_constant(operands[2]), 6) & 0x3F
                op_val = 2 if opcode == "ldb" else 3
                machine_word = (op_val << 12) | (reg << 9) | (base << 6) | offset6

            elif opcode in {"lshf", "rshfl", "rshfa"}:
                dr = parse_register(operands[0])
                sr = parse_register(operands[1])
                amount = parse_constant(operands[2])
                if opcode == "lshf":
                    mode = 0b00
                elif opcode == "rshfl":
                    mode = 0b01
                else:
                    mode = 0b11
                machine_word = encode_shift(13, dr, sr, mode, amount)

            elif opcode == "rti":
                machine_word = 0x8000

            elif opcode == "trap":
                machine_word = encode_trap(operands[0])

            elif opcode == "halt":
                machine_word = 0xF025

            elif opcode == "nop":
                machine_word = 0x0000

            else:
                raise AssemblyError(f"Unhandled opcode {opcode}")

            if machine_word is not None:
                out.write(f"0x{machine_word & 0xFFFF:04X}\n")
                current_address += 2


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print("Usage: python assembler.py <source.asm> <output.obj>")
        sys.exit(1)

    source_path, output_path = argv[1], argv[2]
    with open(source_path, "r", encoding="utf-8") as asm_file:
        lines = asm_file.readlines()

    try:
        symbols, _ = build_symbol_table(lines)
        assemble(lines, symbols, output_path)
    except AssemblyError as err:
        print(f"Assembly failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
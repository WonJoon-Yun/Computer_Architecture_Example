import sys
import os
import argparse
from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser
from io import StringIO

def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(description="LC-3b assembler")
    parser.add_argument("source_directory", type=str, nargs="?", default="program", help="source directory containing assembly files")
    parser.add_argument("result_directory", type=str, nargs="?", default="results", help="output directory for assembled files")
    parser.add_argument("--program", type=str, default=".asm", help="extension of the source files (default: .asm)")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--debug", action="store_true", help="enable detailed debugging output")
    return parser.parse_args()

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


class DebugLogger:
    """Logger that writes to both console and file."""
    def __init__(self, debug_file: Optional[str] = None, console: bool = True):
        self.console = console
        self.debug_file = debug_file
        self.file_handle = None
        if debug_file:
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            self.file_handle = open(debug_file, "w", encoding="utf-8")
    
    def print(self, *args, **kwargs):
        """Print to both console and file if enabled."""
        message = " ".join(str(arg) for arg in args)
        if self.console:
            print(*args, **kwargs)
        if self.file_handle:
            self.file_handle.write(message + "\n")
            self.file_handle.flush()
    
    def close(self):
        """Close the debug file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# --------------------------- Utility helpers ---------------------------

# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None

def set_debug_logger(logger: Optional[DebugLogger]):
    """Set the global debug logger."""
    global _debug_logger
    _debug_logger = logger

def debug_print(*args, **kwargs):
    """Print debug message using the global logger if available."""
    if _debug_logger:
        _debug_logger.print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def tokenize(line: str, debug: bool = False, line_num: int = 0) -> List[str]:
    """Split a line into tokens, removing comments and commas."""
    original_line = line.rstrip()
    code = line.split(";", 1)[0]
    comment = line.split(";", 1)[1] if ";" in line else ""
    code = code.replace(",", " ")
    tokens = [tok for tok in code.strip().split() if tok]
    if debug and tokens:
        debug_print(f"  [Line {line_num:3d}] Tokenize: '{original_line}'")
        debug_print(f"    -> Tokens: {tokens}")
        if comment:
            debug_print(f"    -> Comment: '{comment.strip()}'")
    return tokens


def is_register(token: str) -> bool:
    return token.lower().startswith(REGISTER_PREFIX) and token[1:].isdigit() and 0 <= int(token[1:]) <= 7


def parse_register(token: str, debug: bool = False) -> int:
    if not is_register(token):
        raise AssemblyError(f"Invalid register {token}")
    reg_num = int(token[1:])
    if debug:
        debug_print(f"    -> Parse register: {token} -> R{reg_num}")
    return reg_num


def parse_constant(token: str, debug: bool = False) -> int:
    """Parse a numeric constant of the form #number or xHEX, with optional sign."""
    value = None
    if token.startswith("#"):
        value = int(token[1:], 10)
        if debug:
            debug_print(f"    -> Parse constant (decimal): {token} -> {value} (0x{value:04X})")
    elif token.startswith("x"):
        # Support x-FF syntax where the sign follows the x prefix.
        sign = 1
        digits = token[1:]
        if digits.startswith("-"):
            sign = -1
            digits = digits[1:]
        value = sign * int(digits, 16)
        if debug:
            debug_print(f"    -> Parse constant (hex): {token} -> {value} (0x{value & 0xFFFF:04X})")
    else:
        raise AssemblyError(f"Invalid constant {token}")
    return value


def sign_extend(value: int, bits: int, debug: bool = False) -> int:
    original = value
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= (1 << bits)
    if debug and original != value:
        debug_print(f"    -> Sign extend: {original} -> {value} ({bits} bits)")
    return value


# --------------------------- Symbol table pass ---------------------------

def build_symbol_table(lines: List[str], debug: bool = False) -> Tuple[Dict[str, int], int]:
    """
    First pass: build a symbol table mapping labels to addresses.
    Returns the table and the starting address from .ORIG.
    """
    symbols: Dict[str, int] = {}
    address = None
    started = False

    if debug:
        debug_print("\n=== PASS 1: Building Symbol Table ===")

    for line_num, raw in enumerate(lines, 1):
        tokens = tokenize(raw.lower(), debug, line_num)
        if not tokens:
            continue
        if not started:
            if tokens[0] == ".orig":
                if len(tokens) < 2:
                    raise AssemblyError(".orig requires an address")
                address = parse_constant(tokens[1], debug)
                started = True
                if debug:
                    debug_print(f"  [Line {line_num:3d}] .ORIG: Starting address = 0x{address:04X} ({address})")
            continue

        # Stop processing after .end
        if tokens[0] == ".end":
            if debug:
                debug_print(f"  [Line {line_num:3d}] .END: End of assembly")
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
            if debug:
                debug_print(f"  [Line {line_num:3d}] Label '{label}' -> 0x{address:04X} ({address})")

        if opcode_index >= len(tokens):
            continue

        opcode = tokens[opcode_index]
        if opcode == ".fill" or opcode in OPCODES:
            if debug:
                debug_print(f"  [Line {line_num:3d}] Address 0x{address:04X}: {opcode} (size: 2 bytes)")
            address += 2
        # Other pseudo-ops (.end) handled above

    if address is None:
        raise AssemblyError("No .orig found in source")
    
    if debug:
        debug_print(f"\nSymbol Table ({len(symbols)} symbols):")
        for sym, addr in sorted(symbols.items()):
            debug_print(f"  {sym:20s} -> 0x{addr:04X} ({addr})")
        debug_print(f"Final address: 0x{address:04X} ({address})")
        debug_print("=" * 50)
    
    return symbols, address


# --------------------------- Encoding helpers ---------------------------

def pc_offset(target: int, current_address: int, bits: int, debug: bool = False) -> int:
    offset = (target - (current_address + 2)) // 2
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    if debug:
        debug_print(f"    -> PC offset: target=0x{target:04X}, current=0x{current_address:04X}, offset={offset} ({bits} bits)")
    if offset < min_val or offset > max_val:
        raise AssemblyError(f"Offset {offset} out of range for {bits} bits (range: {min_val} to {max_val})")
    masked = offset & ((1 << bits) - 1)
    if debug:
        debug_print(f"    -> Masked offset: 0x{masked:04X} (binary: {masked:0{bits}b})")
    return masked


def encode_trap(vector_token: str, debug: bool = False) -> int:
    trap_vector = parse_constant(vector_token, debug)
    if trap_vector < 0 or trap_vector > 0xFF:
        raise AssemblyError("Trap vector out of range")
    encoded = 0xF000 | trap_vector
    if debug:
        debug_print(f"    -> TRAP encoding: vector={trap_vector} -> 0x{encoded:04X}")
    return encoded


def encode_shift(opcode: int, dr: int, sr: int, mode_bits: int, amount: int, debug: bool = False) -> int:
    if not (0 <= amount <= 15):
        raise AssemblyError("Shift amount must be 0-15")
    encoded = (opcode << 12) | (dr << 9) | (sr << 6) | (mode_bits << 4) | amount
    if debug:
        debug_print(f"    -> Shift encoding: op={opcode:04b}, dr=R{dr}, sr=R{sr}, mode={mode_bits:02b}, amount={amount} -> 0x{encoded:04X}")
    return encoded


# --------------------------- Second pass ---------------------------

def assemble(lines: List[str], symbols: Dict[str, int], output_path: str, debug: bool = False) -> None:
    started = False
    current_address = 0

    if debug:
        debug_print("\n=== PASS 2: Code Generation ===")

    with open(output_path, "w", encoding="utf-8") as out:
        for line_num, raw in enumerate(lines, 1):
            tokens = tokenize(raw.lower(), debug, line_num)
            if not tokens:
                continue
            if not started:
                if tokens[0] == ".orig":
                    current_address = parse_constant(tokens[1], debug)
                    out.write(f"0x{current_address:04X}\n")
                    started = True
                    if debug:
                        debug_print(f"  [Line {line_num:3d}] Write .ORIG: 0x{current_address:04X}")
                continue

            if tokens[0] == ".end":
                if debug:
                    debug_print(f"  [Line {line_num:3d}] .END: Stopping assembly")
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

            if debug:
                debug_print(f"  [Line {line_num:3d}] Address 0x{current_address:04X}: {opcode} {operands}")

            machine_word = None

            if opcode == ".fill":
                value_token = operands[0]
                if value_token in symbols:
                    value = symbols[value_token]
                    if debug:
                        debug_print(f"    -> .FILL: Symbol '{value_token}' -> 0x{value:04X}")
                else:
                    value = parse_constant(value_token, debug)
                machine_word = value & 0xFFFF
                if debug:
                    debug_print(f"    -> .FILL encoding: 0x{machine_word:04X}")

            elif opcode.startswith("br"):
                cond_str = opcode[2:]
                n = 1 if ("n" in cond_str or cond_str == "") else 0
                z = 1 if ("z" in cond_str or cond_str == "") else 0
                p = 1 if ("p" in cond_str or cond_str == "") else 0
                if cond_str == "":
                    n = z = p = 1
                target_token = operands[0]
                if target_token in symbols:
                    target = symbols[target_token]
                    if debug:
                        debug_print(f"    -> BR target: Symbol '{target_token}' -> 0x{target:04X}")
                else:
                    target = parse_constant(target_token, debug)
                offset = pc_offset(target, current_address, 9, debug)
                machine_word = (0 << 12) | (n << 11) | (z << 10) | (p << 9) | offset
                if debug:
                    debug_print(f"    -> BR encoding: n={n}, z={z}, p={p}, offset={offset} -> 0x{machine_word:04X} ({machine_word:016b})")

            elif opcode == "add" or opcode == "and" or opcode == "xor":
                opcode_val = 1 if opcode == "add" else 5 if opcode == "and" else 9
                dr = parse_register(operands[0], debug)
                sr1 = parse_register(operands[1], debug)
                if is_register(operands[2]):
                    sr2 = parse_register(operands[2], debug)
                    machine_word = (opcode_val << 12) | (dr << 9) | (sr1 << 6) | sr2
                    if debug:
                        debug_print(f"    -> {opcode.upper()} encoding (register): op={opcode_val}, dr=R{dr}, sr1=R{sr1}, sr2=R{sr2} -> 0x{machine_word:04X}")
                else:
                    imm5 = parse_constant(operands[2], debug)
                    imm5 = sign_extend(imm5, 5, debug) & 0x1F
                    machine_word = (opcode_val << 12) | (dr << 9) | (sr1 << 6) | 0x20 | imm5
                    if debug:
                        debug_print(f"    -> {opcode.upper()} encoding (immediate): op={opcode_val}, dr=R{dr}, sr1=R{sr1}, imm5={imm5} -> 0x{machine_word:04X}")

            elif opcode == "not":
                dr = parse_register(operands[0], debug)
                sr = parse_register(operands[1], debug)
                machine_word = (9 << 12) | (dr << 9) | (sr << 6) | 0x3F
                if debug:
                    debug_print(f"    -> NOT encoding: dr=R{dr}, sr=R{sr} -> 0x{machine_word:04X}")

            elif opcode == "jmp":
                base = parse_register(operands[0], debug)
                machine_word = (12 << 12) | (base << 6)
                if debug:
                    debug_print(f"    -> JMP encoding: base=R{base} -> 0x{machine_word:04X}")

            elif opcode == "ret":
                machine_word = (12 << 12) | (7 << 6)
                if debug:
                    debug_print(f"    -> RET encoding: -> 0x{machine_word:04X}")

            elif opcode == "jsr":
                target_token = operands[0]
                if target_token in symbols:
                    target = symbols[target_token]
                    if debug:
                        debug_print(f"    -> JSR target: Symbol '{target_token}' -> 0x{target:04X}")
                else:
                    target = parse_constant(target_token, debug)
                offset = pc_offset(target, current_address, 11, debug)
                machine_word = (4 << 12) | (1 << 11) | offset
                if debug:
                    debug_print(f"    -> JSR encoding: offset={offset} -> 0x{machine_word:04X}")

            elif opcode == "jsrr":
                base = parse_register(operands[0], debug)
                machine_word = (4 << 12) | (base << 6)
                if debug:
                    debug_print(f"    -> JSRR encoding: base=R{base} -> 0x{machine_word:04X}")

            elif opcode == "lea":
                dr = parse_register(operands[0], debug)
                target_token = operands[1]
                if target_token in symbols:
                    target = symbols[target_token]
                    if debug:
                        debug_print(f"    -> LEA target: Symbol '{target_token}' -> 0x{target:04X}")
                else:
                    target = parse_constant(target_token, debug)
                offset = pc_offset(target, current_address, 9, debug)
                machine_word = (14 << 12) | (dr << 9) | offset
                if debug:
                    debug_print(f"    -> LEA encoding: dr=R{dr}, offset={offset} -> 0x{machine_word:04X}")

            elif opcode == "ldw" or opcode == "stw":
                reg = parse_register(operands[0], debug)
                base = parse_register(operands[1], debug)
                offset6 = sign_extend(parse_constant(operands[2], debug), 6, debug) & 0x3F
                op_val = 6 if opcode == "ldw" else 7
                machine_word = (op_val << 12) | (reg << 9) | (base << 6) | offset6
                if debug:
                    print(f"    -> {opcode.upper()} encoding: reg=R{reg}, base=R{base}, offset6={offset6} -> 0x{machine_word:04X}")

            elif opcode == "ldb" or opcode == "stb":
                reg = parse_register(operands[0], debug)
                base = parse_register(operands[1], debug)
                offset6 = sign_extend(parse_constant(operands[2], debug), 6, debug) & 0x3F
                op_val = 2 if opcode == "ldb" else 3
                machine_word = (op_val << 12) | (reg << 9) | (base << 6) | offset6
                if debug:
                    debug_print(f"    -> {opcode.upper()} encoding: reg=R{reg}, base=R{base}, offset6={offset6} -> 0x{machine_word:04X}")

            elif opcode in {"lshf", "rshfl", "rshfa"}:
                dr = parse_register(operands[0], debug)
                sr = parse_register(operands[1], debug)
                amount = parse_constant(operands[2], debug)
                if opcode == "lshf":
                    mode = 0b00
                elif opcode == "rshfl":
                    mode = 0b01
                else:
                    mode = 0b11
                machine_word = encode_shift(13, dr, sr, mode, amount, debug)

            elif opcode == "rti":
                machine_word = 0x8000
                if debug:
                    debug_print(f"    -> RTI encoding: -> 0x{machine_word:04X}")

            elif opcode == "trap":
                machine_word = encode_trap(operands[0], debug)

            elif opcode == "halt":
                machine_word = 0xF025
                if debug:
                    debug_print(f"    -> HALT encoding: -> 0x{machine_word:04X}")

            elif opcode == "nop":
                machine_word = 0x0000
                if debug:
                    debug_print(f"    -> NOP encoding: -> 0x{machine_word:04X}")

            else:
                raise AssemblyError(f"Unhandled opcode {opcode}")

            if machine_word is not None:
                out.write(f"0x{machine_word & 0xFFFF:04X}\n")
                if debug:
                    debug_print(f"    -> Write: 0x{machine_word:04X} (binary: {machine_word:016b})")
                current_address += 2


def assemble_file(source_path: str, output_path: str, verbose: bool = False, debug: bool = False, debug_dir: Optional[str] = None) -> bool:
    """Assemble a single source file to an output file."""
    debug_logger = None
    try:
        # Set up debug logging to file if debug is enabled
        if debug and debug_dir:
            debug_file = os.path.join(debug_dir, os.path.basename(source_path) + ".debug")
            debug_logger = DebugLogger(debug_file, console=True)
            set_debug_logger(debug_logger)
        
        if debug or verbose:
            debug_print(f"\n{'='*60}")
            debug_print(f"Assembling: {source_path}")
            debug_print(f"Output: {output_path}")
            if debug and debug_dir:
                debug_print(f"Debug log: {debug_logger.debug_file if debug_logger else 'N/A'}")
            debug_print(f"{'='*60}")
        
        with open(source_path, "r", encoding="utf-8") as asm_file:
            lines = asm_file.readlines()

        symbols, _ = build_symbol_table(lines, debug)
        assemble(lines, symbols, output_path, debug)
        
        if debug:
            debug_print(f"\n{'='*60}")
            debug_print(f"Assembly complete: {source_path} -> {output_path}")
            if debug_dir:
                debug_print(f"Debug log saved to: {debug_logger.debug_file if debug_logger else 'N/A'}")
            debug_print(f"{'='*60}\n")
        elif verbose:
            print(f"Assembled {source_path} -> {output_path}")
        return True
    except AssemblyError as err:
        error_msg = f"Assembly failed for {source_path}: {err}"
        print(error_msg)
        if debug_logger:
            debug_logger.print(error_msg)
            import traceback
            debug_logger.print(traceback.format_exc())
        elif debug:
            import traceback
            traceback.print_exc()
        return False
    except Exception as err:
        error_msg = f"Error processing {source_path}: {err}"
        print(error_msg)
        if debug_logger:
            debug_logger.print(error_msg)
            import traceback
            debug_logger.print(traceback.format_exc())
        elif debug:
            import traceback
            traceback.print_exc()
        return False
    finally:
        if debug_logger:
            debug_logger.close()
            set_debug_logger(None)


def main(source_dir: str, result_dir: str, file_ext: str, verbose: bool = False, debug: bool = False) -> None:
    """Process all assembly files in the source directory."""
    # Ensure directories exist
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    # Create result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Create debug directory if debug is enabled
    debug_dir = None
    if debug:
        # Create debug directory at the same level as result_dir
        result_abs = os.path.abspath(result_dir)
        base_dir = os.path.dirname(result_abs)
        debug_dir = os.path.join(base_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        if verbose:
            print(f"Debug logs will be saved to: {debug_dir}")
    
    # Normalize file extension (ensure it starts with a dot)
    # If user passed a filename like "example.asm", extract just the extension
    original_ext = file_ext
    if "." in file_ext and not file_ext.startswith("."):
        # Check if it looks like a filename (has characters before the dot)
        parts = file_ext.split(".", 1)
        if len(parts) == 2 and parts[0]:  # Has a name part before the extension
            # Extract just the extension part
            file_ext = "." + parts[1]
            if verbose:
                print(f"Note: Extracted extension '{file_ext}' from '{original_ext}'")
    elif not file_ext.startswith("."):
        file_ext = "." + file_ext
    
    # Find all files with the specified extension
    asm_files = [f for f in os.listdir(source_dir) if f.endswith(file_ext)]
    
    if not asm_files:
        print(f"No files with extension '{file_ext}' found in '{source_dir}'")
        sys.exit(1)
    
    # Process each file
    failed_count = 0
    for asm_file in sorted(asm_files):
        source_path = os.path.join(source_dir, asm_file)
        # Change extension to .obj for output
        output_file = os.path.splitext(asm_file)[0] + ".obj"
        output_path = os.path.join(result_dir, output_file)
        
        if not assemble_file(source_path, output_path, verbose, debug, debug_dir):
            failed_count += 1
    
    if failed_count > 0:
        print(f"\nFailed to assemble {failed_count} file(s)")
        sys.exit(1)
    elif verbose:
        print(f"\nSuccessfully assembled {len(asm_files)} file(s)")


if __name__ == "__main__":
    args = parse_args()
    if args.verbose or args.debug:
        print(f"Processing files from '{args.source_directory}' to '{args.result_directory}'")
        if args.debug:
            print("Debug mode: ENABLED")
    main(args.source_directory, args.result_directory, args.program, args.verbose, args.debug)
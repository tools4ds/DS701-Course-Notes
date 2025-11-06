#!/usr/bin/env python3
"""
Strip Quarto markdown content based on profile visibility tags.

This script removes content blocks that are not visible for the selected profile
(web or slides) and strips {.fragment} and {.incremental} tags.
"""

import argparse
import re
import sys
from pathlib import Path


def check_special_block(div_attrs, profile):
    """
    Check if a content block has special handling (conditional, fragment, or incremental).
    
    Args:
        div_attrs: The attributes string from a div tag (e.g., '{.content-visible when-profile="web"}')
        profile: The selected profile ('web' or 'slides')
    
    Returns:
        Tuple of (is_special, should_remove_content, should_remove_wrapper)
        - is_special: True if this block has special handling
        - should_remove_content: True if the content should be removed
        - should_remove_wrapper: True if the div wrapper tags should be removed
    """
    # Check for fragment or incremental - remove wrapper, keep content
    if re.search(r'\.fragment\b', div_attrs) or re.search(r'\.incremental\b', div_attrs):
        return (True, False, True)
    
    # Check for content-visible
    visible_match = re.search(r'\.content-visible\s+when-profile="(\w+)"', div_attrs)
    if visible_match:
        target_profile = visible_match.group(1)
        if target_profile != profile:
            # Content not visible for this profile - remove everything
            return (True, True, True)
        else:
            # Content visible for this profile - keep content but remove wrapper
            return (True, False, True)
    
    # Check for content-hidden
    hidden_match = re.search(r'\.content-hidden\s+when-profile="(\w+)"', div_attrs)
    if hidden_match:
        target_profile = hidden_match.group(1)
        if target_profile == profile:
            # Content hidden for this profile - remove everything
            return (True, True, True)
        else:
            # Content not hidden for this profile - keep content but remove wrapper
            return (True, False, True)
    
    return (False, False, False)


def process_file(input_file, profile):
    """
    Process a Quarto markdown file and remove content based on profile.
    
    Args:
        input_file: Path to the input file
        profile: The selected profile ('web' or 'slides')
    
    Returns:
        List of lines to write to output
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    output_lines = []
    # Stack of (colon_count, should_skip_content, should_skip_wrapper) tuples
    skip_stack = []
    
    for line_num, line in enumerate(lines, 1):
        # Check for div markers (:::+ with optional attributes)
        # First check if it matches the div pattern
        div_match = re.match(r'^(\s*)(:{3,})(\s+\{[^}]+\})?\s*$', line)
        
        if div_match:
            indent = div_match.group(1)
            colons = div_match.group(2)
            attrs = div_match.group(3) or ''
            colon_count = len(colons)
            
            # Determine if this is an opening or closing div
            # It's a closing div if: no attributes AND we have something on the stack with matching colons
            is_closing = not attrs and skip_stack and any(cc == colon_count for cc, _, _ in skip_stack)
            
            if is_closing:
                # Handle closing div
                # Find matching opening div
                found_match = False
                while skip_stack:
                    open_colon_count, skip_content, skip_wrapper = skip_stack[-1]
                    skip_stack.pop()
                    
                    if open_colon_count == colon_count:
                        # Found matching opening div
                        found_match = True
                        if skip_wrapper:
                            # Skip this closing div too
                            pass
                        else:
                            # Keep this closing div
                            output_lines.append(line)
                        break
                    elif open_colon_count < colon_count:
                        # Mismatch - put it back and treat as regular line
                        skip_stack.append((open_colon_count, skip_content, skip_wrapper))
                        if not any(sc for _, sc, _ in skip_stack):
                            output_lines.append(line)
                        break
                
                # If we've exhausted the stack without finding a match, treat as regular line
                if not found_match and not skip_stack:
                    output_lines.append(line)
            else:
                # Handle opening div
                currently_skipping = any(skip_content for _, skip_content, _ in skip_stack)
                
                if attrs and not currently_skipping:
                    # Check if this block has special handling
                    is_special, should_remove_content, should_remove_wrapper = check_special_block(attrs, profile)
                    
                    if is_special:
                        # Track this special block
                        skip_stack.append((colon_count, should_remove_content, should_remove_wrapper))
                        # Don't output the opening tag if we're removing the wrapper
                        if not should_remove_wrapper:
                            # This shouldn't happen for our use cases, but keep it for safety
                            output_lines.append(line)
                    else:
                        # Normal block - keep it
                        skip_stack.append((colon_count, False, False))
                        output_lines.append(line)
                elif currently_skipping:
                    # We're inside a skip block, track nesting and skip
                    skip_stack.append((colon_count, True, True))
                else:
                    # No attributes, track the div and output it
                    skip_stack.append((colon_count, False, False))
                    output_lines.append(line)
            continue
        
        # Regular line - add to output if not skipping content
        currently_skipping = any(skip_content for _, skip_content, _ in skip_stack)
        if not currently_skipping:
            output_lines.append(line)
    
    return output_lines


def main():
    parser = argparse.ArgumentParser(
        description='Strip Quarto markdown content based on profile visibility tags.'
    )
    parser.add_argument('input_file', help='Input Quarto markdown file (.qmd)')
    parser.add_argument('--profile', required=True, choices=['web', 'slides'],
                        help='Profile to use for visibility filtering')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    if input_path.suffix != '.qmd':
        print(f"Warning: File '{args.input_file}' does not have .qmd extension.", file=sys.stderr)
    
    # Generate output filename
    output_path = input_path.parent / f"{input_path.stem}-stripped{input_path.suffix}"
    
    # Process the file
    try:
        output_lines = process_file(input_path, args.profile)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        print(f"Processed file written to: {output_path}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()


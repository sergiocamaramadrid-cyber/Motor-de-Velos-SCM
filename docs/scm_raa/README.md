# SCM-RAA (v3.0 - experimental)

Reflexive diagnostic layer for CRTT.

CRTT detects structure.  
RAA evaluates if that detection is reliable.

## Example

SPARC → WEAK (moderate)  
LITTLE THINGS → FAILURE (low)  
YANG → STRONG (high)

## Status

Experimental. Not used in v2.5.x results.

## Purpose

Avoid false positives and over-interpretation of weak transitions.

## Decision layer

RAA outputs are converted into information layers:

- `foreground` → main analysis
- `midground` → directed exploration
- `background` → control/reference

The framework does not discard data.  
It organizes datasets by information content.

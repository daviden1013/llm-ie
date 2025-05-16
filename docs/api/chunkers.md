# Chunkers API

This module provides classes for splitting documents into manageable units for processing by LLMs and for providing context to those units.

## Unit Chunkers

Unit chunkers determine how a document is divided into smaller pieces for frame extraction. Each piece is a `FrameExtractionUnit`.

::: llm_ie.chunkers.UnitChunker
::: llm_ie.chunkers.WholeDocumentUnitChunker
::: llm_ie.chunkers.SentenceUnitChunker
::: llm_ie.chunkers.TextLineUnitChunker

## Context Chunkers

Context chunkers determine what contextual information is provided to the LLM alongside a specific `FrameExtractionUnit`.

::: llm_ie.chunkers.ContextChunker
::: llm_ie.chunkers.NoContextChunker
::: llm_ie.chunkers.WholeDocumentContextChunker
::: llm_ie.chunkers.SlideWindowContextChunker
# An Automatic Question Answer Generator for Japanese Texts 

This is a simple approach to automatically generate QA from Japanese texts.
No machine learning technique is used here, but for some complicated use cases, statistical methods maybe better choices.

This approach contains the following steps:
- Dependency Parsing
- Semantic Analysis (Japanese case  grammar analysis) to generate semantic labels for chunks.
- Generate QA using dependencies and semantic labels

Note that I did not implement all patterns for semantic label generation.
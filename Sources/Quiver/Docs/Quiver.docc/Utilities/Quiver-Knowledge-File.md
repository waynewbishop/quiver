# Quiver Knowledge File

Pairing the Quiver reference with Claude as a study companion for the math and code.

## Overview

The Quiver knowledge file is a special Markdown document that captures Quiver's API surface, design principles, and validation status in one place. Uploaded into a conversation with Claude, it becomes a study companion — a place to ask questions about the math behind a method, the meaning of a result, or the reasoning behind a Quiver design choice. The book, the cookbook, and the documentation are where Claude is taught.

### What Claude is

Claude is an AI assistant from [Anthropic](https://www.anthropic.com). It answers questions in plain language. When given a reference document — like the Quiver knowledge file — it grounds its answers in that document, which keeps responses accurate to the framework as it actually exists rather than to a general impression of how Swift numerical libraries usually work.

### How knowledge files work

Claude accepts uploaded files as context for a conversation. A markdown file dropped into a chat window is read, indexed, and used as the source of truth for any question that follows. Two patterns are common:

- **One-off conversation** — drag the knowledge file into a fresh chat, ask a question, get an answer. The context lives for the duration of that conversation.
- **Claude Project** — for longer study sessions or a full course, create a Project, attach the knowledge file once, and every conversation inside that Project reads the same reference automatically. This is the recommended pattern for anyone working through the cookbook or a primer over multiple sittings.

In both cases, the assistant's answers cite the parts of the file it draws from, so a learner can cross-check against the framework's actual API rather than trusting a paraphrase.

### Obtaining the file

The knowledge file lives in the main Quiver repository. Clone the repo and the file is in the `Docs/` folder:

```bash
git clone https://github.com/waynewbishop/quiver.git
```

The file is at `Docs/quiver-knowledge.md`. A new copy ships with each Quiver release, so re-cloning or pulling the latest version keeps the reference aligned with the API and the other framework utilities.

### Where the companion fits

The knowledge file works alongside the primary learning surfaces, each playing to its strength: the <doc:Quiver-Notebook> for runnable code, the <doc:Quiver-Cookbook> for worked recipes, and the documentation for method signatures. The companion answers questions across all three. A learner stuck on a primer paragraph, unsure why a recipe produced a particular number, or wanting a worked example phrased a different way can bring those questions over and get a grounded explanation that points back to the primary resource.

### Privacy and what gets sent

The knowledge file and any questions a learner types are sent to Claude's servers for processing. This is different from the <doc:Quiver-Notebook>, which runs entirely on the local machine and sends nothing across the network. For classroom labs, exam settings, or air-gapped environments, the Notebook is the appropriate surface; the knowledge file pattern requires an internet connection and acceptance of the [Anthropic terms of service](https://www.anthropic.com/legal/consumer-terms).

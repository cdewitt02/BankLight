# Thoughts:

## What it does well:
- SQL RAG is happening
- Integrates with DropBox web hook
- Ok RAG classification

## Needs:
- Better Category classifier
    - Might need to manually create a model that incrementally gets better
- Get vector search working
- Better RAG classification
- Better system prompts
    - These have gotten better, added chat history into context
- Need a bigger and better model
    - switch to lambda3.1 8b
- Need SVT

## Stretch:
- Run dropbox listener on RPI and sync with model server's db
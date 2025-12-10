# Data Model: AI-Driven Technical Book

## Entities

### Technical Book
- **Attributes**:
  - title: string
  - description: string
  - author: string
  - chapters: array of Chapter references
  - wordCount: integer (8,000-12,000 total)
  - status: enum (draft, in-review, published)
  - createdAt: datetime
  - updatedAt: datetime
  - innovativeElements: array of InteractiveComponent references

### Chapter
- **Attributes**:
  - id: string (unique identifier)
  - title: string
  - content: string (Markdown/MDX format)
  - wordCount: integer
  - position: integer (chapter order)
  - status: enum (draft, in-review, published)
  - codeExamples: array of CodeExample references
  - citations: array of Citation references
  - interactiveComponents: array of InteractiveComponent references
  - createdAt: datetime
  - updatedAt: datetime

### CodeExample
- **Attributes**:
  - id: string (unique identifier)
  - title: string
  - description: string
  - code: string (the actual code snippet)
  - language: string (programming language)
  - runnable: boolean
  - tested: boolean
  - interactive: boolean (for innovative live examples)
  - createdAt: datetime
  - updatedAt: datetime

### InteractiveComponent
- **Attributes**:
  - id: string (unique identifier)
  - type: enum (live-editor, interactive-diagram, simulation, visualization, quiz)
  - title: string
  - description: string
  - componentCode: string (React component code)
  - props: object (configuration for the component)
  - createdAt: datetime
  - updatedAt: datetime

### Citation
- **Attributes**:
  - id: string (unique identifier)
  - title: string
  - url: string (URL to source)
  - sourceType: enum (documentation, article, official, other)
  - verified: boolean
  - createdAt: datetime

### DocusaurusSite
- **Attributes**:
  - id: string (unique identifier)
  - title: string
  - description: string
  - config: object (Docusaurus configuration)
  - sidebar: object (navigation structure)
  - theme: string
  - innovativeFeatures: array of string (features like search, dark mode, etc.)
  - createdAt: datetime
  - updatedAt: datetime

### Deployment
- **Attributes**:
  - id: string (unique identifier)
  - target: enum (github-pages, netlify, vercel)
  - status: enum (pending, success, failed)
  - url: string (deployment URL)
  - deployedAt: datetime
  - commitHash: string
  - size: integer (size of the build in bytes)
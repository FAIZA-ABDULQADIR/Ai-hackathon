# ADR-001: Docusaurus-GitHub Pages-MDX Technology Stack

## Status
Accepted

## Date
2025-12-07

## Context
We need to select a technology stack for creating an innovative AI-driven technical book with 8-12 chapters and 8,000-12,000 words. The solution must support interactive elements, have excellent performance, and enable cutting-edge technical documentation with AI-assisted content generation. The team needs a solution that supports rich interactive components and can scale to the full scope of the book.

## Decision
We will use the Docusaurus + GitHub Pages + MDX technology stack, which includes:
- **Framework**: Docusaurus 3.x as the static site generator
- **Deployment**: GitHub Pages as the primary hosting solution
- **Format**: MDX (Markdown + React components) for content creation
- **Supporting tools**: Node.js 18+, npm/yarn package manager

## Consequences

### Positive
- Enables rich interactive components through React integration
- Supports the full scope of 8-12 chapters and 8,000-12,000 words
- Excellent plugin ecosystem and extensibility for innovative features
- Superior support for MDX allowing embedded React components
- Advanced search capabilities with potential Algolia integration
- Strong adoption in the developer community for cutting-edge documentation
- Cost-effective deployment through GitHub Pages
- Tight integration with GitHub workflow for collaborative development
- Supports AI-assisted content generation workflows

### Negative
- Learning curve for MDX and React components
- More complex than pure Markdown solutions
- Potential performance challenges with rich interactive content
- Larger bundle sizes due to React components
- Requires more sophisticated build processes

## Alternatives
- **MkDocs + GitHub Pages**: Simpler but limited customization options, primarily Python-based, lacks innovation potential for interactive elements
- **Hugo + Netlify**: Fast build times but requires learning Go templating, less interactive features, limited for innovative content
- **Custom React App + Vercel**: More control but significantly more development overhead, no built-in documentation features
- **GitBook**: Good documentation features but less customization for innovative interactive elements

## References
- plan.md
- research.md
- data-model.md
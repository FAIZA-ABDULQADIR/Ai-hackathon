# Interactive AI-Driven Physical AI & Humanoid Robotics Book

This project contains an innovative AI-assisted technical book focused on Physical AI, embodied intelligence, and humanoid robotics built with Docusaurus and deployed to GitHub Pages.

## About This Book

This technical book provides a comprehensive guide to Physical AI and humanoid robotics, covering everything from fundamental ROS 2 concepts to advanced NVIDIA Isaac platforms and Vision-Language-Action systems. The book follows a 13-week curriculum organized into 5 progressive modules, with interactive elements and runnable code examples.

## Features

- Interactive code examples with live editors
- Performance optimized for fast loading
- WCAG 2.1 AA accessibility compliance
- Mobile-responsive design
- Search functionality
- GitHub Pages deployment

## Prerequisites

- Node.js 18+
- npm or yarn package manager
- Git for version control

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-textbook
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

This will start the Docusaurus development server at `http://localhost:3000`.

## Directory Structure

```
.
├── docs/                    # Book content (chapters, sections)
│   ├── intro.md             # Introduction chapter
│   ├── chapter-1/           # Module 1: ROS 2 - Robotic Nervous System
│   │   ├── index.md         # Chapter main content
│   │   └── code-examples/   # Chapter-specific code examples
│   ├── chapter-2/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-3/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-4/           # Module 2: Digital Twin - Gazebo + Unity
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-5/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-6/           # Module 3: NVIDIA Isaac - Perception, VSLAM, Nav2
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-7/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-8/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-9/           # Module 4: Vision-Language-Action - Whisper → LLM → ROS
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-10/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-11/
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-12/          # Module 5: Humanoid Robotics & Capstone
│   │   ├── index.md
│   │   └── code-examples/
│   ├── chapter-13/
│   │   ├── index.md
│   │   └── code-examples/
├── src/                     # Custom React components and pages
│   ├── components/          # Reusable React components
│   └── pages/               # Custom pages
├── static/                  # Static assets (images, files)
│   ├── img/                 # Images and graphics
│   └── files/               # Downloadable files
├── docusaurus.config.js     # Docusaurus configuration
├── sidebars.js              # Navigation sidebar configuration
└── package.json             # Project dependencies and scripts
```

## Building and Deployment

To build the complete book:

```bash
npm run build
```

To serve the built site locally:

```bash
npm run serve
```

Deployment to GitHub Pages happens automatically via GitHub Actions when changes are pushed to the main branch.

## AI-Assisted Workflow

This book was created using an AI-assisted workflow with:
- Spec-Kit Plus for structured content creation
- Claude Code for technical content generation and validation
- Human oversight for quality assurance and accuracy verification
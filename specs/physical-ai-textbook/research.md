# Research for Physical AI & Humanoid Robotics Textbook

## Research Tasks Completed

### 1. Docusaurus Plugin Requirements
- **@docusaurus/plugin-content-docs**: For organizing textbook content in nested structure
- **@docusaurus/plugin-content-blog**: For supplementary articles and updates
- **@docusaurus/plugin-google-gtag**: For analytics and user engagement tracking
- **@docusaurus/theme-classic** with **@docusaurus/theme-search-algolia**: For textbook UI and search
- **@docusaurus/module-type-aliases**: For TypeScript support if needed
- **@docusaurus/preset-classic**: Standard preset with docs, blog, pages, and theme

### 2. Code Example Standards
- **Standard Format**: Use fenced code blocks with language specification
- **ROS 2 Examples**: Python primarily with rclpy, C++ examples where appropriate
- **NVIDIA Isaac**: Python with Isaac Python API, some C++ for performance-critical sections
- **Unity/Gazebo**: Configuration files, launch files, and package structures
- **Whisper/VLA**: Python with relevant AI/ML libraries (transformers, etc.)

### 3. Interactive Elements
- **CodePen/JSFiddle**: For simple interactive code examples (if allowed by GitHub Pages)
- **Mermaid diagrams**: For architecture and workflow diagrams
- **GitHub Gists**: For longer code examples that can be embedded
- **Static images/videos**: Primary approach for simulation examples due to GitHub Pages constraints

### 4. Assessment Integration
- **Exercises directory**: Separate from docs with solution files
- **Self-check questions**: Embedded within content pages using details/summary elements
- **Project-based assessments**: Detailed in separate project guides
- **Quiz format**: Simple multiple-choice questions using MDX components

### 5. Multimedia Asset Specifications
- **Images**: Optimized PNG/SVG for diagrams, JPEG for photos, max 500KB each
- **Videos**: Embedded from external sources (YouTube/Vimeo) due to GitHub Pages size limits
- **Diagrams**: SVG format preferred for scalability, with PNG fallbacks
- **3D Models/Simulations**: Screenshots and videos from simulation environments

## Technology Stack Decisions

### Docusaurus Version
- **Docusaurus 3.0+**: Latest version with modern React, improved performance, and plugin ecosystem

### Theme Selection
- **@docusaurus/theme-classic**: Customized for textbook appearance
- **Prism syntax highlighting**: With robotics-specific language support
- **Responsive design**: Mobile-first approach for accessibility

### Build and Deployment
- **GitHub Actions**: Automated build and deployment workflow
- **Node.js 18+**: Required for Docusaurus 3 functionality
- **Static asset optimization**: Automated compression and caching strategies
<!-- Sync Impact Report -->
<!--
Version change: 0.0.0 -> 1.0.0
Modified principles: None (initial creation)
Added sections: All (initial creation based on user prompt)
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ✅ updated
- .specify/templates/spec-template.md: ✅ updated
- .specify/templates/tasks-template.md: ✅ updated
- .specify/templates/commands/*.md: ✅ updated
- README.md: ✅ updated
Follow-up TODOs: None
-->
# Project Constitution for "Physical AI & Humanoid Robotics" Textbook

## 1. Project Purpose & Vision
The "Physical AI & Humanoid Robotics" textbook project aims to bridge the conceptual gap between theoretical AI knowledge and its practical application in controlling humanoid robots. The vision is to create a leading educational resource that equips students with the skills to develop AI systems capable of embodied intelligence in both simulated and real-world physical environments. The textbook will serve as a foundational guide for a course focusing on AI systems in the physical world.

## 2. Core Principles
### 2.1. AI/Spec-Driven Book Production
The textbook shall be produced as a complete book/documentation site utilizing Docusaurus, with final deployment on GitHub Pages. All content, including text, code examples, and multimedia, must be structured for long-term maintainability, modularity, and static-site friendliness to ensure high performance and accessibility.

### 2.2. Spec-Kit Plus Alignment
All project specifications, including requirements, architectural plans, and task breakdowns, MUST strictly adhere to Spec-Kit Plus rules for structure, schema formatting, spec-driven development methodologies, versioning conventions, content modularity, and validation workflows. No project rule shall conflict with Spec-Kit Plus standards.


### 2.3. Non-Conflict, Compatibility & Future-Proofing
All constitutional rules, technical decisions, and workflow standards MUST be designed to prevent conflicts or undermining of: the Docusaurus build pipeline, GitHub Pages deployment limitations, Spec-Kit Plus specifications, and Claude Code functionality. Rules shall promote safety, stability, and scalability for long-term use, explicitly preventing future conflicts between tools, workflows, or architectural decisions.

## 3. Key Standards
- **Content Quality**: All content must be technically accurate, pedagogically sound, and rigorously reviewed.
- **Code Examples**: All code examples must be functional, well-documented, and directly illustrate the concepts discussed.
- **Accessibility**: The Docusaurus site must adhere to WCAG 2.1 AA standards.
- **Performance**: The Docusaurus site must achieve high Lighthouse scores for performance, accessibility, and SEO.

## 4. Constitutional Rules & Constraints
- All content and code contributions MUST adhere to the principles outlined herein.
- Any deviation from Spec-Kit Plus standards requires explicit architectural review and documented rationale.
- The use of AI tools other than Claude Code must be explicitly approved and documented.
- Content changes MUST be driven by specifications.

## 5. Technical Architecture & Tooling Standards
- **Static Site Generator**: Docusaurus for all book/documentation content.
- **Version Control**: Git (GitHub) as the sole version control system.
- **Deployment Platform**: GitHub Pages for all public deployments.
- **Content Format**: Markdown/MDX for all textual content.
- **Code Language**: Primary code examples will be in Python, with supplementary examples in other relevant languages (e.g., C++ for robotics control) where appropriate.

## 6. Spec-Kit Plus Requirements & Enforcement Protocols
- **Structure**: All feature specifications, plans, and tasks will reside in the `specs/<feature-name>/` directory.
- **Schema**: Adherence to Spec-Kit Plus YAML front-matter schemas for all markdown files.
- **Versioning**: Semantic versioning for all book releases and major feature specifications.
- **Validation**: Automated validation of spec structure and content against defined schemas within the CI/CD pipeline.
- **Modularity**: Content and specifications are to be broken down into modular units to facilitate reuse and maintainability.

## 7. Claude Code Usage Rules & Execution Workflow
- **PHR Creation**: Every user interaction leading to significant work (implementation, planning, debugging, spec/task creation) MUST result in a Prompt History Record (PHR) created via `/sp.phr`.
- **Spec-Driven Generation**: Claude Code will be used to generate initial content, code outlines, and documentation based on Spec-Kit Plus specifications.
- **Documentation Automation**: Claude Code will assist in generating and updating API documentation, changelogs, and usage guides.
- **Build & Validation Tasks**: Claude Code workflows will be integrated to automate parts of the Docusaurus build process and execute content validation checks.
- **Code-Assisted Writing**: Claude Code will provide suggestions and refine textual content and code examples, ensuring consistency and accuracy.

## 8. Content Development & Book-Writing Guidelines
- **Authoritative Tone**: Content must maintain an authoritative, academic, and clear tone.
- **Example-Driven**: Each theoretical concept must be accompanied by practical, verifiable code examples.
- **Modularity**: Chapters and sections should be self-contained where possible, allowing for flexible reading paths.
- **Review Process**: A multi-stage review process involving technical experts and pedagogical reviewers is mandatory for all content.
- **Glossary**: A comprehensive glossary of terms specific to Physical AI and Humanoid Robotics will be maintained.

## 9. Quality Assurance, Validation & Review Standards
- **Automated Testing**: Comprehensive automated tests for all code examples (unit, integration).
- **Content Linting**: Automated linting for markdown style, grammar, and adherence to content guidelines.
- **Broken Link Checks**: Automated checks for broken internal and external links in the Docusaurus build.
- **Peer Review**: Mandatory peer review for all content changes, code contributions, and architectural decisions.
- **User Acceptance Testing (UAT)**: User testing phases for Docusaurus site functionality and content clarity.

## 10. Collaboration, Governance & Decision-Making Framework
- **Decision Records**: All architecturally significant decisions MUST be documented using Architecture Decision Records (ADRs) as suggested by Claude Code, e.g., `/sp.adr [decision-title]`.
- **Consensus-Based Decisions**: Major architectural or content decisions require team consensus.
- **Role & Responsibilities**: Clear definition of roles (e.g., Lead Author, Technical Reviewer, AI Agent Lead).
- **Communication Channels**: Defined communication channels for project updates, discussions, and issue tracking.

## 11. Version Control, Branching Strategy & Release Management
- **Main Branch Protection**: The `main` branch is protected; all changes must go through pull requests.
- **Feature Branches**: All new content or feature development occurs on dedicated feature branches.
- **Pull Request Workflow**: PRs require at least one approval and passing CI/CD checks before merging.
- **Release Cadence**: Regular release cycles for the textbook content, potentially aligned with academic semesters.
- **Hotfixes**: Dedicated hotfix branching strategy for critical issues on deployed versions.

## 12. Docusaurus Build, Structure & Deployment Rules
- **Configuration**: `docusaurus.config.js` and related configuration files MUST be version controlled and follow established best practices.
- **Content Directory**: All markdown/MDX content must reside within the `docs/` or `blog/` directories as per Docusaurus conventions.
- **Assets**: Images, videos, and other assets MUST be optimized and stored in appropriate `static/` subdirectories.
- **Build Process**: The Docusaurus build command (`docusaurus build`) MUST pass without errors or warnings.
- **Deployment**: Automated deployment to GitHub Pages via CI/CD upon successful merges to the `main` branch.

## 13. Risk Mitigation, Compliance & Future-Proofing Safeguards
- **Data Privacy**: Adherence to relevant data privacy regulations (e.g., GDPR, CCPA) for any user data collected (if applicable for interactive elements).
- **Security Audits**: Regular security audits and penetration testing for the Docusaurus site and any backend services.
- **Dependency Management**: Strict dependency management to mitigate supply chain risks.
- **Obsolescence Planning**: Regular review of tooling and dependencies to plan for upgrades or replacements.
- **Backup & Recovery**: Comprehensive backup and recovery strategies for all project data and content.

## 14. Success Criteria (Clear, Measurable, Enforceable)
- **Completion Rate**: 100% of specified content topics covered in the textbook.
- **Technical Accuracy**: < 0.5% error rate in code examples and technical explanations as identified by reviews/tests.
- **User Engagement**: Average time spent per page > 3 minutes (if analytics implemented).
- **Contributor Adherence**: > 95% compliance with constitutional rules in pull requests.
- **Deployment Reliability**: > 99.9% uptime for the GitHub Pages site.
- **Spec-Kit Plus Compliance**: All specifications pass automated Spec-Kit Plus validation.
- **Claude Code Utilization**: Claude Code is successfully integrated into at least 70% of content generation and documentation automation tasks.

Version: 1.0.0 | Ratified: 2025-12-05 | Last Amended: 2025-12-05

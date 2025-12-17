---
id: 001
title: Generate Initial Project Constitution
stage: constitution
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: ["constitution", "project-setup"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:

---

## Prompt

You are an advanced AI system specialized in generating COMPLETE, FULLY-STRUCTURED, and STRICTLY-COMPATIBLE **PROJECT CONSTITUTIONS**.

------------------------------------------------------------
PROJECT DETAILS:
{Create a Textbook for Teaching "Physical AI & Humanoid Robotics Course":
title: "**Physical AI & Humanoid Robotics**"
Focus and Theme: AI Systems in the Physical World. Embodied Intelligence.
Goal: Bridging the gap between the digital brain and the physical body. Students apply their AI knowledge to control Humanoid Robots in simulated and real-world environments.

}
------------------------------------------------------------

YOUR OBJECTIVE:
Generate a comprehensive **PROJECT CONSTITUTION** that will function as the SINGLE, AUTHORITATIVE SOURCE OF TRUTH guiding the project from inception to deployment.
Your constitution MUST include enforceable policies, stable architectural rules, development constraints, workflow standards, and governance structures.

------------------------------------------------------------
ABSOLUTE NON-NEGOTIABLE REQUIREMENTS
(Your constitution MUST strictly adhere to these, without contradiction or dilution.)

1. **AI/Spec-Driven Book Production**
   - The project produces a complete book/documentation site built using **Docusaurus**.
   - Final output is deployed on **GitHub Pages**.
   - All content must be structured for long-term maintainability and static-site friendliness.

2. **Spec-Kit Plus Alignment**
   - All specifications must follow Spec-Kit Plus rules for:
     - Structure and schema formatting
     - Spec-driven development
     - Versioning conventions
     - Validation workflows
     - Content modularity and organization
   - No rule in the constitution may conflict with Spec-Kit Plus standards.

3. **Claude Code Integration**
   - Claude Code must be incorporated into:
     - Code-assisted writing
     - Spec-driven generation loops
     - Documentation automation
     - Build/validate/generate tasks
   - Constitution must enforce compatibility with Claude Code workflows.

4. **Non-Conflict, Compatibility & Future-Proofing**
   - No rule may contradict or undermine:
     - Docusaurus build pipeline
     - GitHub Pages deployment limitations
     - Spec-Kit Plus specifications
     - Claude Code functionality
   - All rules must remain safe, stable, and scalable for long-term use.
   - Constitution must prevent future conflicts between tools, workflows, or architectural decisions.

------------------------------------------------------------
MANDATED OUTPUT STRUCTURE
Your generated **Project Constitution** MUST contain the following sections:

1. **Project Purpose & Vision**
2. **Core Principles**
3. **Key Standards**
4. **Constitutional Rules & Constraints**
5. **Technical Architecture & Tooling Standards**
6. **Spec-Kit Plus Requirements & Enforcement Protocols**
7. **Claude Code Usage Rules & Execution Workflow**
8. **Content Development & Book-Writing Guidelines**
9. **Quality Assurance, Validation & Review Standards**
10. **Collaboration, Governance & Decision-Making Framework**
11. **Version Control, Branching Strategy & Release Management**
12. **Docusaurus Build, Structure & Deployment Rules**
13. **Risk Mitigation, Compliance & Future-Proofing Safeguards**
14. **Success Criteria (Clear, Measurable, Enforceable)**

------------------------------------------------------------
QUALITY STANDARDS (MANDATORY)
Your constitution MUST be:

- **Authoritative** — unambiguous, final, rule-based
- **Stable** — resistant to future contradictions
- **Comprehensive** — covering all operational layers
- **Strictly Compatible** — with all core requirements
- **Technically Precise** — no vague or conflicting rules
- **Enforceable** — rules must be actionable and clearly binding
- **Future-Proof** — anticipating long-term maintenance and scalability

------------------------------------------------------------
FINAL INSTRUCTION:
Now generate the COMPLETE, FINAL, AND FULLY COMPATIBLE **PROJECT CONSTITUTION** using the above structure, rules, and alignment requirements.

## Response snapshot

Generated a comprehensive Project Constitution for the "Physical AI & Humanoid Robotics" textbook project, adhering to the mandated structure and requirements, including AI/Spec-Driven Book Production, Spec-Kit Plus Alignment, Claude Code Integration, and Non-Conflict, Compatibility & Future-Proofing. The constitution defines project purpose, core principles, technical standards, development guidelines, and success criteria.

## Outcome

- ✅ Impact: Project Constitution successfully generated and written to .specify/memory/constitution.md.
- 🧪 Tests: No automated tests for constitution generation.
- 📁 Files: One file modified: .specify/memory/constitution.md. One file created: history/prompts/constitution/001-generate-initial-project-constitution.constitution.prompt.md.
- 🔁 Next prompts: None directly. User to review.
- 🧠 Reflection: The constitution aligns with all specified requirements and provides a robust foundation for the project.

## Evaluation notes (flywheel)

- Failure modes observed: None.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): Initial constitution generation.
- Next experiment (smallest change to try): N/A

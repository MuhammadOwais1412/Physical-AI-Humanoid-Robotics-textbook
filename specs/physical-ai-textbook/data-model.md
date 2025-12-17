# Data Model for Physical AI & Humanoid Robotics Textbook

## Content Entities

### Module
- **id**: Unique identifier for the module (e.g., "module-1-ros2")
- **title**: Display title of the module
- **description**: Brief overview of the module content
- **learningObjectives**: Array of learning objectives for the module
- **duration**: Estimated time to complete (in weeks)
- **prerequisites**: Array of prerequisite knowledge or modules
- **sections**: Array of section IDs that belong to this module

### Section
- **id**: Unique identifier for the section
- **title**: Display title of the section
- **module**: Reference to parent module ID
- **order**: Position within the module
- **content**: Path to the content file
- **learningObjectives**: Specific learning objectives for this section
- **codeExamples**: Array of code example IDs
- **exercises**: Array of exercise IDs
- **assets**: Array of asset file paths

### ContentPage
- **id**: Unique identifier for the content page
- **title**: Page title
- **section**: Reference to parent section ID
- **content**: Markdown/MDX content
- **metadata**: Additional page metadata (tags, difficulty, time to read)
- **relatedPages**: Array of related content page IDs
- **prerequisites**: Array of prerequisite content page IDs

### CodeExample
- **id**: Unique identifier for the code example
- **title**: Title/description of the example
- **language**: Programming language (python, c++, etc.)
- **module**: Reference to the module this example belongs to
- **section**: Reference to the section this example belongs to
- **code**: The actual code content
- **explanation**: Text explaining the code
- **dependencies**: List of required packages/libraries
- **output**: Expected output or behavior

### Exercise
- **id**: Unique identifier for the exercise
- **title**: Exercise title
- **module**: Reference to parent module ID
- **section**: Reference to parent section ID
- **type**: Type of exercise (practical, theoretical, coding, etc.)
- **difficulty**: Difficulty level (beginner, intermediate, advanced)
- **instructions**: Detailed instructions for the exercise
- **solution**: Solution to the exercise (may be stored separately)
- **evaluationCriteria**: How the exercise will be evaluated

### Asset
- **id**: Unique identifier for the asset
- **filename**: Name of the file
- **path**: Path to the asset file
- **type**: Type of asset (image, video, diagram, model, etc.)
- **description**: Brief description of the asset
- **usage**: Where the asset is used (module, section, page references)
- **altText**: Alternative text for accessibility

## Relationships

### Module → Section
- One-to-many: One module contains many sections
- Module has an array of section references

### Section → ContentPage
- One-to-many: One section contains many content pages
- Section has an array of content page references

### Section → CodeExample
- One-to-many: One section has many code examples
- Section has an array of code example references

### Section → Exercise
- One-to-many: One section has many exercises
- Section has an array of exercise references

### ContentPage → Asset
- Many-to-many: Content pages can reference multiple assets, assets can be used in multiple pages
- Content page has array of asset references

## Validation Rules

### Module Validation
- Title must be 5-100 characters
- Description must be 10-500 characters
- Learning objectives must be 3-10 items
- Duration must be between 1-4 weeks
- Sections must be ordered sequentially

### Section Validation
- Title must be 5-80 characters
- Order must be a positive integer
- Content path must exist and be valid
- Learning objectives must align with parent module

### ContentPage Validation
- Title must match the first H1 in content
- Content must contain at least 200 characters
- Metadata must include difficulty and time to read
- Related pages must be valid content page IDs

### CodeExample Validation
- Code must be syntactically valid for the specified language
- Language must be in the supported list (python, c++, bash, etc.)
- Explanation must be provided for all code examples
- Dependencies must be documented

### Exercise Validation
- Type must be one of: practical, theoretical, coding, research
- Difficulty must be one of: beginner, intermediate, advanced
- Instructions must be clear and complete
- Solution must be provided for evaluation

## State Transitions

### Content Development Workflow
1. **Draft**: Content is being created/edited
2. **Review**: Content is under peer review
3. **Approved**: Content has passed review and is ready for publication
4. **Published**: Content is live in the textbook
5. **Archived**: Content is outdated but preserved for reference

### Validation States
- **Validated**: Content has passed all automated validation checks
- **Needs Revision**: Content failed validation and requires changes
- **Pending**: Content awaiting validation
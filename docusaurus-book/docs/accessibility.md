# Accessibility Guidelines

This document outlines the accessibility standards and testing procedures for the Physical AI & Humanoid Robotics textbook to ensure it meets WCAG 2.1 AA compliance.

## Web Content Accessibility Guidelines (WCAG) 2.1 AA Compliance

Our textbook implementation follows the Web Content Accessibility Guidelines (WCAG) 2.1 AA standards to ensure equal access for all learners, including those with disabilities.

### Perceivable

1. **Text Alternatives**: All images, diagrams, and visual content include descriptive alt text.
2. **Time-based Media**: Video and audio content includes captions and transcripts where applicable.
3. **Adaptable Content**: Content can be presented in different ways without losing information or structure.
4. **Distinguishable**: Foreground and background colors have sufficient contrast (at least 4.5:1 for normal text).

### Operable

1. **Keyboard Accessible**: All functionality is available from a keyboard.
2. **Enough Time**: Users have enough time to read and use content.
3. **Seizures and Physical Reactions**: Content does not cause seizures or physical reactions.
4. **Navigable**: Users can navigate, find content, and determine where they are.

### Understandable

1. **Readable**: Text content is readable and understandable.
2. **Predictable**: Web pages appear and operate in predictable ways.
3. **Input Assistance**: Users are helped to avoid and correct mistakes.

### Robust

1. **Compatible**: Content is robust enough to be interpreted reliably by various assistive technologies.

## Accessibility Features Implemented

### 1. Semantic HTML Structure
- Proper heading hierarchy (H1, H2, H3, etc.)
- Use of semantic elements (nav, main, section, article)
- Landmark roles for screen readers

### 2. Color and Contrast
- Minimum 4.5:1 contrast ratio for normal text
- 3:1 contrast ratio for large text
- Color not used as the only visual means of conveying information

### 3. Navigation
- Skip links for keyboard users
- Logical tab order
- Clear focus indicators
- Breadcrumb navigation

### 4. Forms and Inputs
- Proper labels for all form controls
- Clear error messages and instructions
- Accessible validation

## Testing Procedures

### Automated Testing

1. **WAVE Evaluation Tool**: Use the WAVE browser extension to identify accessibility issues.
2. **axe DevTools**: Run axe-core tests to catch potential accessibility problems.
3. **Lighthouse**: Use Lighthouse in Chrome DevTools to audit accessibility.

### Manual Testing

1. **Keyboard Navigation**: Navigate the entire site using only the Tab key.
2. **Screen Reader Testing**: Test with screen readers like NVDA, JAWS, or VoiceOver.
3. **Color Contrast**: Verify contrast ratios using tools like the Color Contrast Analyzer.
4. **Focus Indicators**: Ensure all interactive elements have visible focus indicators.

### Example Testing Commands

```bash
# Run accessibility audit with Lighthouse CLI
npx lighthouse http://localhost:3000 --only-categories=accessibility

# Run axe-core tests (if integrated)
npm run test:accessibility
```

## Specific Considerations for Robotics Content

### Diagrams and Visual Content
- All technical diagrams include detailed alt text descriptions
- Complex diagrams are broken down into simpler components when possible
- Interactive diagrams include keyboard navigation alternatives

### Code Examples
- Code examples have sufficient color contrast
- Syntax highlighting is not the only means of conveying information
- Code examples include text descriptions of functionality

### Videos and Simulations
- Technical videos include captions and audio descriptions
- Interactive simulations provide alternative text-based explanations
- Time-based interactions have adjustable timing options

## Common Accessibility Issues and Solutions

### Issue: Low Contrast Text
**Solution**: Use tools like the WebAIM Contrast Checker to ensure proper contrast ratios.

### Issue: Missing Alt Text
**Solution**: Provide concise, descriptive alt text for all meaningful images. Decorative images should have empty alt attributes.

### Issue: Poor Keyboard Navigation
**Solution**: Ensure all interactive elements are focusable and the tab order follows a logical sequence.

### Issue: Inaccessible Forms
**Solution**: Always associate labels with form controls and provide clear error messages.

## Authoring Practices

### Writing for Accessibility
1. Use clear, simple language
2. Provide descriptive link text
3. Use proper heading structure
4. Include alternative text for images
5. Use captions and summaries for tables

### Technical Implementation
1. Use ARIA labels and roles appropriately
2. Ensure responsive design works across devices
3. Test with various assistive technologies
4. Validate HTML markup

## Accessibility Checklist

Before publishing new content, ensure:

- [ ] All images have appropriate alt text
- [ ] Headings follow proper hierarchy
- [ ] Links have descriptive text
- [ ] Color contrast meets WCAG guidelines
- [ ] Content is keyboard navigable
- [ ] Forms have proper labels
- [ ] Videos have captions
- [ ] Interactive elements have focus indicators
- [ ] No content flashes more than 3 times per second

## Tools for Maintaining Accessibility

- **WAVE**: Web Accessibility Evaluation Tool
- **axe**: Accessibility testing engine
- **Color Contrast Analyzer**: For checking color contrast ratios
- **NVDA**: Free screen reader for testing
- **Lighthouse**: Built-in accessibility auditing

## Reporting Accessibility Issues

If you encounter accessibility issues with the textbook:

1. Document the specific problem and location
2. Note the assistive technology used (if applicable)
3. Describe the expected vs. actual behavior
4. Submit feedback through the appropriate channel

## Ongoing Maintenance

Accessibility is not a one-time effort. Regular maintenance includes:

1. Regular automated accessibility scans
2. Periodic manual testing
3. Updates to meet evolving standards
4. User feedback incorporation
5. Review of new content before publication

This accessibility guide ensures that our Physical AI & Humanoid Robotics textbook remains accessible to all learners, regardless of their abilities or the assistive technologies they use.
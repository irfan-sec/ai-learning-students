# Images Directory

This directory contains visual assets for the AI Learning Students GitHub Pages site.

## 📁 Organization

```
images/
├── README.md           # This file
├── logos/              # Course and institution logos
├── diagrams/           # Technical diagrams and illustrations
├── screenshots/        # Interface and demo screenshots
└── icons/              # UI icons and symbols
```

## 📐 Image Guidelines

### Format Recommendations
- **PNG**: For diagrams, screenshots, and images requiring transparency
- **JPG**: For photographs and complex images
- **SVG**: For icons, logos, and simple graphics (preferred when possible)

### Size Guidelines
- **Optimize for web**: Keep file sizes under 1MB when possible
- **Responsive sizes**: Provide multiple sizes if needed
- **Consider retina displays**: 2x versions for high-DPI screens

### Naming Convention
Use descriptive, kebab-case filenames:
- `week01-intelligent-agents-diagram.png`
- `search-algorithms-comparison.jpg`
- `ai-ethics-framework.svg`

### Accessibility
- Always include meaningful alt text in markdown
- Use high contrast for diagrams and text overlays
- Consider colorblind-friendly color palettes

## 🖼️ Usage in Pages

Reference images using relative paths:
```markdown
![AI Agent Architecture](images/diagrams/agent-architecture.png)
```

For responsive images:
```html
<img src="images/diagrams/search-tree.png" 
     alt="Search tree visualization" 
     class="responsive-image">
```

## 📄 Attribution

When using external images:
- Include proper attribution
- Ensure licensing allows educational use
- Document sources in commit messages

## 🎨 Visual Style

Maintain consistency with the site theme:
- Use the course color palette when possible
- Keep diagrams clean and minimalist
- Ensure text is readable at various sizes
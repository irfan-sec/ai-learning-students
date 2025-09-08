# Contributing to AI Learning Students

We welcome contributions from students, educators, and the AI community! This repository is designed to be a collaborative educational resource that grows and improves with community input.

## 🎯 Ways to Contribute

### 1. **Content Improvements**
- Fix typos, grammar, or formatting issues
- Improve explanations in notes and exercises
- Add clearer examples or analogies
- Update outdated links or resources

### 2. **New Resources**
- Share relevant articles, videos, or tutorials
- Add links to interactive demos or visualizations
- Suggest additional textbook references
- Contribute useful online tools or datasets

### 3. **Code Contributions**
- Submit Python implementations of algorithms
- Create Jupyter notebooks with step-by-step examples
- Add visualization scripts or interactive demos
- Optimize existing code for clarity or performance

### 4. **Visual Content**
- Create or suggest better diagrams and illustrations
- Design flowcharts for complex algorithms
- Add screenshots of tools or interfaces
- Contribute infographics or concept maps

### 5. **Practice Materials**
- Submit additional exercise questions
- Create project ideas and assignments
- Add quiz questions with explanations
- Share real-world application examples

---

## 📋 Contribution Guidelines

### Before You Start
1. **Check existing issues** to see if someone else is working on similar improvements
2. **Browse the repository** to understand the structure and content style
3. **Review recent contributions** to maintain consistency

### Content Standards
- **Educational Focus:** All content should serve the learning objectives
- **Beginner-Friendly:** Assume readers are 3rd-semester undergraduates
- **Clear Structure:** Use consistent headings, bullet points, and formatting
- **Accurate Information:** Verify facts and cite reliable sources
- **Python Focus:** Code examples should primarily use Python 3.x

### File Structure Guidelines

The repository now supports both the original research structure and a GitHub Pages documentation site:

#### Original Research Structure
```
weeks/
├── WeekXX_Topic_Name/
│   ├── notes.md          # Core concepts and explanations
│   ├── resources.md      # Links and references
│   ├── exercises.md      # Practice questions
│   ├── code/             # Python scripts and notebooks
│   └── images/           # Diagrams and illustrations
```

#### GitHub Pages Documentation Structure
```
docs/
├── _config.yml           # Jekyll configuration
├── index.md             # Course overview and navigation
├── weekXX.md            # Individual week pages (week01.md, week02.md, etc.)
└── images/              # Visual assets for the documentation site
    └── README.md        # Image documentation
```

---

## 🔧 How to Contribute

### Step 1: Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/ai-learning-students.git
cd ai-learning-students
```

### Step 2: Choose Your Contribution Type

#### Option A: Documentation Site (GitHub Pages)
For improvements to the main documentation site:
- Edit files in the `/docs` directory
- Follow the existing week format (weekXX.md)
- Test locally with Jekyll if possible: `bundle exec jekyll serve`
- Ensure all links are relative for GitHub Pages compatibility

#### Option B: Research Materials
For detailed course materials and exercises:
- Edit files in the `/weeks` directory
- Maintain the existing folder structure
- Include comprehensive exercises and code examples

### Step 3: Create a Feature Branch
```bash
git checkout -b feature/your-contribution-name
# Examples: feature/week3-astar-examples, fix/week7-typos, docs/improve-navigation
```

### Step 4: Make Your Changes
- Follow the existing file structure and naming conventions
- For GitHub Pages contributions: ensure Jekyll compatibility
- For research materials: maintain comprehensive educational content
- Test any code contributions to ensure they work
- Preview markdown files to check formatting

### Step 5: Commit and Push
```bash
git add .
git commit -m "Add: Clear description of your changes"
git push origin feature/your-contribution-name
```

### Step 6: Create a Pull Request
- Use a descriptive title and detailed description
- Specify whether changes affect GitHub Pages docs or research materials
- Reference any related issues
- Include screenshots for visual changes

---

## 📝 Content Style Guide

### Writing Style
- **Conversational but Professional:** Write as if explaining to a friend
- **Active Voice:** "We implement the algorithm" vs "The algorithm is implemented"
- **Consistent Terminology:** Use the same terms as the AIMA textbook when possible
- **Code Comments:** Explain what the code does, not just how

### Markdown Formatting
- Use `#` for main headings, `##` for sections, `###` for subsections
- Include code blocks with language specification: ```python
- Use bullet points for lists and numbered lists for sequential steps
- Add alt text for all images: `![Description](image-url)`

### Code Standards
- **PEP 8 Compliance:** Follow Python style guidelines
- **Meaningful Names:** Use descriptive variable and function names
- **Documentation:** Include docstrings for functions and classes
- **Error Handling:** Add basic error checking where appropriate

---

## 🎨 Visual Content Guidelines

### Images and Diagrams
- **Format:** PNG or JPG for static images, GIF for simple animations
- **Size:** Optimize images to be under 1MB when possible
- **Attribution:** Always credit the source of images
- **Alt Text:** Include descriptive alt text for accessibility
- **GitHub Pages:** Place images in `/docs/images/` and reference with relative paths
- **Research Materials:** Place images in `/weeks/WeekXX_Topic/images/`

### Suggested Visual Content
- Algorithm flowcharts and pseudocode diagrams
- Search tree visualizations
- Neural network architectures
- Data flow diagrams
- Concept maps linking related topics

---

## ⚡ Quick Contribution Ideas

### Easy (Good for beginners)
- Fix typos or broken links in documentation
- Add missing alt text to images
- Improve code comments
- Update GitHub Pages navigation
- Add relevant external resources

### Medium (Some AI knowledge helpful)
- Add practice exercises for existing weeks
- Create simple Python examples for documentation
- Write clearer explanations of complex concepts
- Improve GitHub Pages formatting and styling
- Convert research materials to documentation format

### Advanced (Requires AI expertise)
- Implement new algorithm examples with full explanations
- Create comprehensive Jupyter notebooks
- Design interactive visualizations
- Develop project assignments
- Add new weeks or major content sections

---

## 🤝 Community Guidelines

### Be Respectful
- Provide constructive feedback
- Be patient with learners at all levels
- Credit others' work appropriately
- Follow academic integrity principles

### Collaborate Effectively
- Communicate clearly in issues and PRs
- Ask questions when unsure
- Share knowledge generously
- Help review others' contributions

---

## 📞 Getting Help

- **Create an Issue:** For questions, bugs, or feature requests
- **Start a Discussion:** For broader topics or collaboration ideas
- **Contact Maintainers:** For urgent matters or repository access

---

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributor section
- Individual week folders for significant contributions
- Special thanks in course presentations or papers

---

Thank you for helping make AI education more accessible and engaging for everyone! 🚀

**Remember:** Every contribution, no matter how small, makes a difference in someone's learning journey.
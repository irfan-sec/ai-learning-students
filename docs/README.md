# AI Learning Students - GitHub Pages Site

This directory contains the Jekyll-powered GitHub Pages site for the AI Learning Students course. The site provides a modern, student-friendly interface to access course materials.

## 🏗️ Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── _layouts/
│   ├── default.html      # Main site template
│   ├── home.html         # Homepage layout
│   └── week.html         # Individual week pages
├── _includes/
│   └── week-navigation.html # Navigation component
├── _sass/
│   └── main.scss         # Main theme styles
├── assets/
│   ├── css/style.scss    # Compiled styles
│   └── js/theme.js       # Interactive functionality
├── index.md             # Homepage content
├── week01.md            # Individual week pages
├── week02.md
├── ...
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## 🎨 Theme Features

- **Modern Design**: Clean, engaging visual design with animations
- **Responsive**: Mobile-first design that works on all devices
- **Interactive**: Smooth navigation and dynamic elements
- **Accessible**: Follows web accessibility best practices
- **Fast**: Optimized for performance

## 🚀 Local Development

To run the site locally:

1. Install Ruby and Bundler
2. Run `bundle install` to install dependencies
3. Run `bundle exec jekyll serve` to start the development server
4. Visit `http://localhost:4000` to view the site

## 📱 GitHub Pages Setup

This site is configured to be served from the `/docs` directory on GitHub Pages. To enable:

1. Go to repository Settings → Pages
2. Set source to "Deploy from a branch"
3. Select "main" branch and "/docs" folder
4. Save changes

The site will be available at: `https://irfan-sec.github.io/ai-learning-students/`

## 🔗 Course Materials

The site links to detailed course materials in the `/weeks` directory of the main repository. Each week page provides:

- Learning objectives and key concepts
- Links to notes, resources, and exercises
- Interactive navigation between weeks
- Direct access to GitHub materials

## 🛠️ Customization

The theme is built with CSS custom properties (variables) for easy customization. Main customization points:

- **Colors**: Edit variables in `_sass/main.scss`
- **Content**: Modify week pages and `index.md`
- **Structure**: Adjust layouts in `_layouts/`
- **Styling**: Update Sass files for design changes

## 📄 License

This site follows the same MIT License as the main course repository.
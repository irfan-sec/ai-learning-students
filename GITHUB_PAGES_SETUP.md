# GitHub Pages Setup Instructions

This document provides step-by-step instructions for enabling GitHub Pages for the AI Learning Students repository.

## 🚀 Quick Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** (in the repository navigation)
3. Scroll down to **Pages** in the left sidebar
4. Under **Source**, select:
   - **Deploy from a branch**
   - **Branch**: `main` (or your default branch)
   - **Folder**: `/docs`
5. Click **Save**

### 2. Wait for Deployment

- GitHub will automatically build and deploy your site
- This may take a few minutes on the first deployment
- You'll see a green checkmark when deployment is complete

### 3. Access Your Site

Your site will be available at:
```
https://irfan-sec.github.io/ai-learning-students/
```

## 🔧 Configuration Details

### Repository Settings
The site is configured to be served from the `/docs` directory with the following settings:

- **Base URL**: `/ai-learning-students` (configured in `_config.yml`)
- **Site URL**: `https://irfan-sec.github.io`
- **Theme**: Custom theme based on minima
- **Plugins**: jekyll-feed, jekyll-sitemap

### DNS and Custom Domains (Optional)
If you want to use a custom domain:

1. Add a `CNAME` file to the `/docs` directory with your domain name
2. Configure DNS records to point to GitHub Pages
3. Update the `url` in `_config.yml` to your custom domain

## 🛠️ Local Development

To run the site locally for development:

```bash
cd docs/
bundle install
bundle exec jekyll serve
```

Then visit `http://localhost:4000/ai-learning-students/`

## 🔄 Automatic Updates

The site will automatically rebuild whenever you push changes to the main branch. You can monitor build status in the **Actions** tab of your repository.

## 📝 Troubleshooting

### Common Issues

**Build Failures:**
- Check the Actions tab for error details
- Ensure all markdown files have proper front matter
- Validate YAML syntax in `_config.yml`

**Missing Styles:**
- Verify the `assets/css/style.scss` file has front matter (`---`)
- Check that Sass files are properly imported

**Broken Links:**
- Use relative URLs: `{{ '/week01.html' | relative_url }}`
- Test links in local development environment

### Getting Help

- GitHub Pages documentation: https://docs.github.com/pages
- Jekyll documentation: https://jekyllrb.com/docs/
- Repository issues: Create an issue in this repository

## 🎯 What's Next?

After enabling GitHub Pages:

1. **Share the URL** with students and educators
2. **Monitor analytics** using GitHub's insights
3. **Gather feedback** and iterate on the design
4. **Add more content** as course materials are developed
5. **Consider SEO** optimization for better discoverability

## 🔗 Related Files

- `/docs/_config.yml` - Main Jekyll configuration
- `/docs/README.md` - Technical documentation for the docs directory
- `THEME.md` - Theme design documentation
- `CONTRIBUTING.md` - Contribution guidelines

---

*Once GitHub Pages is enabled, the site will provide a modern, accessible interface for the AI Learning Students course materials!*
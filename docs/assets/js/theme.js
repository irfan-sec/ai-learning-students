// AI Learning Students - Interactive Theme Scripts

document.addEventListener('DOMContentLoaded', function() {
  
  // Mobile menu functionality
  const mobileToggle = document.getElementById('mobileToggle');
  const navMenu = document.getElementById('navMenu');
  
  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener('click', function() {
      navMenu.classList.toggle('active');
      mobileToggle.classList.toggle('active');
    });
  }
  
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
  
  // Progress tracking based on scroll
  function updateProgress() {
    const scrolled = window.scrollY;
    const maxHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = Math.min(Math.max(scrolled / maxHeight * 100, 0), 100);
    
    const progressCircle = document.querySelector('.progress-circle');
    const progressText = document.querySelector('.progress-text');
    
    if (progressCircle && progressText) {
      const angle = (progress / 100) * 360;
      progressCircle.style.background = `conic-gradient(var(--primary-color) ${angle}deg, var(--border-light) ${angle}deg)`;
      progressText.textContent = Math.round(progress) + '%';
    }
  }
  
  // Update progress on scroll
  window.addEventListener('scroll', updateProgress);
  updateProgress(); // Initial call
  
  // Add loading animation to images
  const images = document.querySelectorAll('img');
  images.forEach(img => {
    img.addEventListener('load', function() {
      this.style.animation = 'fadeIn 0.5s ease-in';
    });
  });
  
  // Add hover effects to cards and interactive elements
  const cards = document.querySelectorAll('.week-card, .nav-card, .resource-card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-4px)';
    });
    
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
    });
  });
  
  // Animate elements when they come into view
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animation = 'fadeInUp 0.6s ease-out forwards';
        entry.target.style.opacity = '1';
      }
    });
  }, observerOptions);
  
  // Observe elements for animation
  const animatedElements = document.querySelectorAll('h2, h3, .week-card, .learning-objectives, p, ul, ol');
  animatedElements.forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });
  
  // Add typing effect to main heading if present
  const mainHeading = document.querySelector('h1');
  if (mainHeading) {
    const text = mainHeading.textContent;
    mainHeading.textContent = '';
    let i = 0;
    
    function typeWriter() {
      if (i < text.length) {
        mainHeading.textContent += text.charAt(i);
        i++;
        setTimeout(typeWriter, 50);
      }
    }
    
    setTimeout(typeWriter, 500);
  }
  
  // Code block syntax highlighting enhancement
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(block => {
    // Add copy button to code blocks
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy';
    copyBtn.className = 'copy-code-btn';
    copyBtn.style.cssText = `
      position: absolute;
      top: 10px;
      right: 10px;
      background: var(--primary-color);
      color: white;
      border: none;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
      opacity: 0;
      transition: opacity 0.2s;
    `;
    
    const pre = block.parentElement;
    pre.style.position = 'relative';
    pre.appendChild(copyBtn);
    
    pre.addEventListener('mouseenter', () => copyBtn.style.opacity = '1');
    pre.addEventListener('mouseleave', () => copyBtn.style.opacity = '0');
    
    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(block.textContent);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy', 2000);
      } catch (err) {
        console.error('Failed to copy code:', err);
      }
    });
  });
  
  // Enhanced navigation with week completion tracking
  function initWeekTracking() {
    const currentPage = window.location.pathname;
    const weekMatch = currentPage.match(/week(\d+)/);
    
    if (weekMatch) {
      const weekNumber = parseInt(weekMatch[1]);
      
      // Mark previous weeks as completed (simple simulation)
      for (let i = 1; i < weekNumber; i++) {
        const weekLink = document.querySelector(`a[href*="week${i.toString().padStart(2, '0')}"]`);
        if (weekLink) {
          weekLink.classList.add('completed');
          weekLink.insertAdjacentHTML('beforeend', ' <i class="fas fa-check-circle"></i>');
        }
      }
      
      // Mark current week as active
      const currentWeekLink = document.querySelector(`a[href*="week${weekNumber.toString().padStart(2, '0')}"]`);
      if (currentWeekLink) {
        currentWeekLink.classList.add('current');
      }
    }
  }
  
  initWeekTracking();
  
  // Add interactive quiz elements (if quiz sections exist)
  const quizSections = document.querySelectorAll('.quiz, .exercise');
  quizSections.forEach(quiz => {
    quiz.style.cssText = `
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
      border: 1px solid var(--border);
      border-radius: var(--border-radius);
      padding: 1.5rem;
      margin: 1.5rem 0;
      transition: var(--transition);
    `;
    
    quiz.addEventListener('mouseenter', function() {
      this.style.boxShadow = 'var(--shadow-lg)';
      this.style.borderColor = 'var(--primary-color)';
    });
    
    quiz.addEventListener('mouseleave', function() {
      this.style.boxShadow = 'none';
      this.style.borderColor = 'var(--border)';
    });
  });
  
  // Add visual feedback for external links
  const externalLinks = document.querySelectorAll('a[href^="http"]');
  externalLinks.forEach(link => {
    if (!link.querySelector('.fas')) {
      link.insertAdjacentHTML('beforeend', ' <i class="fas fa-external-link-alt" style="font-size: 0.8em; margin-left: 0.25rem;"></i>');
    }
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });
  
  // Performance optimization: Lazy load images
  const lazyImages = document.querySelectorAll('img[data-src]');
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        img.classList.remove('lazy');
        observer.unobserve(img);
      }
    });
  });
  
  lazyImages.forEach(img => imageObserver.observe(img));
  
});
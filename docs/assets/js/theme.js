// AI Learning Students - Enhanced Interactive Theme Scripts

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
  
  // Enhanced Neural Network Interactions
  function initializeNeuralNetwork() {
    const neuralNodes = document.querySelectorAll('.neural-node');
    const connections = document.querySelector('.neural-connections');
    
    if (neuralNodes.length > 0 && connections) {
      // Create connections between layers
      createNeuralConnections(neuralNodes, connections);
      
      // Add click interactions
      neuralNodes.forEach((node, index) => {
        node.addEventListener('click', function() {
          triggerNeuralActivation(node, neuralNodes);
        });
        
        // Add random activation animation
        setTimeout(() => {
          setInterval(() => {
            if (Math.random() > 0.7) {
              node.style.animation = 'neuralPulse 0.5s ease-in-out';
              setTimeout(() => {
                node.style.animation = 'neuralPulse 2s ease-in-out infinite';
              }, 500);
            }
          }, 2000 + Math.random() * 3000);
        }, index * 100);
      });
    }
  }
  
  function createNeuralConnections(nodes, container) {
    const layers = {
      input: document.querySelectorAll('.input-layer .neural-node'),
      hidden1: document.querySelectorAll('.hidden-layer-1 .neural-node'),
      hidden2: document.querySelectorAll('.hidden-layer-2 .neural-node'),
      output: document.querySelectorAll('.output-layer .neural-node')
    };
    
    // Create SVG for connections
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.pointerEvents = 'none';
    
    // Connect input to hidden1
    connectLayers(layers.input, layers.hidden1, svg);
    // Connect hidden1 to hidden2
    connectLayers(layers.hidden1, layers.hidden2, svg);
    // Connect hidden2 to output
    connectLayers(layers.hidden2, layers.output, svg);
    
    container.appendChild(svg);
  }
  
  function connectLayers(layer1, layer2, svg) {
    layer1.forEach(node1 => {
      layer2.forEach(node2 => {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        const rect1 = node1.getBoundingClientRect();
        const rect2 = node2.getBoundingClientRect();
        const containerRect = svg.parentElement.getBoundingClientRect();
        
        line.setAttribute('x1', rect1.left - containerRect.left + rect1.width/2);
        line.setAttribute('y1', rect1.top - containerRect.top + rect1.height/2);
        line.setAttribute('x2', rect2.left - containerRect.left + rect2.width/2);
        line.setAttribute('y2', rect2.top - containerRect.top + rect2.height/2);
        line.setAttribute('stroke', 'rgba(255,255,255,0.2)');
        line.setAttribute('stroke-width', '1');
        
        svg.appendChild(line);
      });
    });
  }
  
  function triggerNeuralActivation(clickedNode, allNodes) {
    // Reset all nodes
    allNodes.forEach(node => {
      node.style.background = 'var(--primary-gradient)';
      node.style.boxShadow = 'none';
    });
    
    // Activate clicked node
    clickedNode.style.background = 'linear-gradient(135deg, #10b981, #06b6d4)';
    clickedNode.style.boxShadow = '0 0 20px rgba(16, 185, 129, 0.8)';
    
    // Create ripple effect
    const ripple = document.createElement('div');
    ripple.style.position = 'absolute';
    ripple.style.width = '100px';
    ripple.style.height = '100px';
    ripple.style.borderRadius = '50%';
    ripple.style.border = '2px solid rgba(16, 185, 129, 0.5)';
    ripple.style.top = '50%';
    ripple.style.left = '50%';
    ripple.style.transform = 'translate(-50%, -50%) scale(0)';
    ripple.style.pointerEvents = 'none';
    ripple.style.animation = 'neuralRipple 1s ease-out forwards';
    
    clickedNode.parentElement.appendChild(ripple);
    
    setTimeout(() => {
      ripple.remove();
    }, 1000);
  }
  
  // Initialize enhanced features
  initializeNeuralNetwork();
  
  // Add loading animation to images with fade-in effect
  const images = document.querySelectorAll('img');
  images.forEach(img => {
    img.addEventListener('load', function() {
      this.style.animation = 'fadeIn 0.8s ease-in';
    });
  });
  
  // Enhanced hover effects for cards and interactive elements
  const cards = document.querySelectorAll('.week-card, .nav-card, .resource-card, .stat-card, .video-card, .preview-section');
  cards.forEach(card => {
    card.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-6px)';
      this.style.transition = 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)';
    });
    
    card.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
    });
  });
  
  // Animate elements when they come into view with enhanced options
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };
  
  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.animation = 'fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1) forwards';
        entry.target.style.opacity = '1';
      }
    });
  }, observerOptions);
  
  // Observe elements for animation
  const animatedElements = document.querySelectorAll('h2, h3, .week-card, .learning-objectives, .stat-card, .preview-section, .video-card, p, ul, ol');
  animatedElements.forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
  });
  
  // Enhanced typing effect for main heading
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
      } else {
        // Add cursor blink effect
        const cursor = document.createElement('span');
        cursor.textContent = '|';
        cursor.style.animation = 'blink 1s infinite';
        mainHeading.appendChild(cursor);
        
        setTimeout(() => {
          cursor.remove();
        }, 3000);
      }
    }
    
    setTimeout(typeWriter, 500);
  }
  
  // Enhanced code block functionality
  const codeBlocks = document.querySelectorAll('pre code');
  codeBlocks.forEach(block => {
    const pre = block.parentElement;
    pre.style.position = 'relative';
    
    // Add copy button to code blocks
    const copyBtn = document.createElement('button');
    copyBtn.textContent = '📋 Copy';
    copyBtn.className = 'copy-code-btn';
    copyBtn.style.cssText = `
      position: absolute;
      top: 10px;
      right: 10px;
      background: var(--primary-color);
      color: white;
      border: none;
      padding: 0.5rem 1rem;
      border-radius: var(--border-radius-sm);
      cursor: pointer;
      font-size: 0.8rem;
      transition: var(--transition);
      z-index: 10;
    `;
    
    copyBtn.addEventListener('click', function() {
      navigator.clipboard.writeText(block.textContent).then(() => {
        this.textContent = '✅ Copied!';
        this.style.background = 'var(--accent-3)';
        setTimeout(() => {
          this.textContent = '📋 Copy';
          this.style.background = 'var(--primary-color)';
        }, 2000);
      });
    });
    
    copyBtn.addEventListener('mouseenter', function() {
      this.style.transform = 'translateY(-2px)';
      this.style.boxShadow = 'var(--shadow)';
    });
    
    copyBtn.addEventListener('mouseleave', function() {
      this.style.transform = 'translateY(0)';
      this.style.boxShadow = 'none';
    });
    
    pre.appendChild(copyBtn);
  });
  
  // Enhanced quiz sections with better interactivity
  const quizSections = document.querySelectorAll('.quiz, .exercise');
  quizSections.forEach(quiz => {
    quiz.style.cssText = `
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
      border: 1px solid var(--border);
      border-radius: var(--border-radius);
      padding: 1.5rem;
      margin: 1.5rem 0;
      transition: var(--transition);
      position: relative;
      overflow: hidden;
    `;
    
    // Add animated border
    const borderGlow = document.createElement('div');
    borderGlow.style.cssText = `
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      background: var(--primary-gradient);
      border-radius: var(--border-radius);
      opacity: 0;
      transition: opacity 0.3s ease;
      z-index: -1;
    `;
    quiz.appendChild(borderGlow);
    
    quiz.addEventListener('mouseenter', function() {
      this.style.boxShadow = 'var(--shadow-lg)';
      this.style.transform = 'translateY(-2px)';
      borderGlow.style.opacity = '0.3';
    });
    
    quiz.addEventListener('mouseleave', function() {
      this.style.boxShadow = 'none';
      this.style.transform = 'translateY(0)';
      borderGlow.style.opacity = '0';
    });
  });
  
  // Enhanced visual feedback for external links
  const externalLinks = document.querySelectorAll('a[href^="http"]');
  externalLinks.forEach(link => {
    if (!link.querySelector('.fas') && !link.querySelector('iframe')) {
      link.insertAdjacentHTML('beforeend', ' <span style="font-size: 0.8em; margin-left: 0.25rem; opacity: 0.7;">🔗</span>');
    }
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
    
    // Add hover animation
    link.addEventListener('mouseenter', function() {
      const icon = this.querySelector('span');
      if (icon) {
        icon.style.transform = 'rotate(45deg) scale(1.2)';
        icon.style.transition = 'transform 0.2s ease';
      }
    });
    
    link.addEventListener('mouseleave', function() {
      const icon = this.querySelector('span');
      if (icon) {
        icon.style.transform = 'rotate(0deg) scale(1)';
      }
    });
  });
  
  // Performance optimization: Enhanced lazy loading for images
  const lazyImages = document.querySelectorAll('img[data-src]');
  const imageObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const img = entry.target;
        img.src = img.dataset.src;
        img.classList.remove('lazy');
        img.classList.add('fade-in');
        observer.unobserve(img);
      }
    });
  });
  
  lazyImages.forEach(img => imageObserver.observe(img));
  
  // Initialize particle system for hero background
  initializeParticleSystem();
  
  function initializeParticleSystem() {
    const heroSection = document.querySelector('.hero-section');
    if (!heroSection) return;
    
    const particleContainer = document.createElement('div');
    particleContainer.className = 'floating-particles';
    particleContainer.style.cssText = `
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 1;
    `;
    
    // Create floating particles
    for (let i = 0; i < 20; i++) {
      const particle = document.createElement('div');
      particle.style.cssText = `
        position: absolute;
        width: ${Math.random() * 4 + 2}px;
        height: ${Math.random() * 4 + 2}px;
        background: rgba(255, 255, 255, ${Math.random() * 0.5 + 0.2});
        border-radius: 50%;
        left: ${Math.random() * 100}%;
        top: ${Math.random() * 100}%;
        animation: floatParticle ${Math.random() * 20 + 10}s ease-in-out infinite;
        animation-delay: ${Math.random() * 5}s;
      `;
      particleContainer.appendChild(particle);
    }
    
    heroSection.appendChild(particleContainer);
  }
  
});

// Add custom animations to CSS
const style = document.createElement('style');
style.textContent = `
  @keyframes neuralRipple {
    0% { transform: translate(-50%, -50%) scale(0); opacity: 1; }
    100% { transform: translate(-50%, -50%) scale(2); opacity: 0; }
  }
  
  @keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
  }
  
  @keyframes floatParticle {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
  }
  
  .fade-in {
    animation: fadeIn 0.8s ease-in-out;
  }
`;
document.head.appendChild(style);
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
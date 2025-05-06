document.addEventListener('DOMContentLoaded', function() {
  // DOM elements
  const dateInput = document.getElementById('date-filter');
  const dateDisplay = document.getElementById('date-display');
  const sessionTypeSelect = document.getElementById('session-type-filter');
  const pitchSelect = document.getElementById('pitch-filter');
  const clearFiltersBtn = document.getElementById('clear-filters');
  const refreshBtn = document.getElementById('refresh-btn');
  const hideFiltersBtn = document.getElementById('hide-filters');
  const filtersContainer = document.getElementById('filters-container');
  const weekTab = document.getElementById('week-tab');
  const monthTab = document.getElementById('month-tab');
  const calendarTitle = document.getElementById('calendar-title');
  const prevBtn = document.getElementById('prev-period');
  const nextBtn = document.getElementById('next-period');
  
  // Populate session types dropdown from API
  async function loadSessionTypes() {
    if (!sessionTypeSelect) return;
    
    try {
      // First add the "All Session Types" option
      const allOption = document.createElement('option');
      allOption.value = "";
      allOption.text = "All Session Types";
      sessionTypeSelect.appendChild(allOption);
      
      // Fetch session types from API
      const response = await fetch('/api/session_types');
      const data = await response.json();
      
      if (data && data.session_types && Array.isArray(data.session_types)) {
        // Then add the specific session types
        data.session_types.forEach(type => {
          const option = document.createElement('option');
          option.value = type;
          option.text = type;
          sessionTypeSelect.appendChild(option);
        });
      } else {
        // Fallback to hardcoded session types if API fails
        const fallbackTypes = [
          'Training',
          'Match',
          'Recovery',
          'Analysis',
          'Fitness Test',
          'Tactical Session'
        ];
        
        fallbackTypes.forEach(type => {
          const option = document.createElement('option');
          option.value = type;
          option.text = type;
          sessionTypeSelect.appendChild(option);
        });
      }
    } catch (error) {
      console.error('Error loading session types:', error);
      
      // Fallback to hardcoded session types
      const fallbackTypes = [
        'Training',
        'Match',
        'Recovery',
        'Analysis',
        'Fitness Test',
        'Tactical Session'
      ];
      
      fallbackTypes.forEach(type => {
        const option = document.createElement('option');
        option.value = type;
        option.text = type;
        sessionTypeSelect.appendChild(option);
      });
    }
  }
  
  // Load session types on page load
  loadSessionTypes();
  
  // Handle date input change
  if (dateInput) {
    dateInput.addEventListener('change', function() {
      if (this.value) {
        const date = new Date(this.value);
        const formattedDate = date.toLocaleDateString('en-US', {
          month: 'long',
          day: 'numeric',
          year: 'numeric'
        });
        
        if (dateDisplay) {
          dateDisplay.textContent = formattedDate;
        }
      }
    });
  }
  
  // Clear filters
  if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', function(e) {
      e.preventDefault();
      
      if (dateInput) dateInput.value = '';
      if (dateDisplay) dateDisplay.textContent = '';
      if (sessionTypeSelect) sessionTypeSelect.value = '';
      if (pitchSelect) pitchSelect.value = '';
    });
  }
  
  // Toggle filters visibility
  if (hideFiltersBtn && filtersContainer) {
    hideFiltersBtn.addEventListener('click', function() {
      const isVisible = filtersContainer.style.display !== 'none';
      
      if (isVisible) {
        filtersContainer.style.display = 'none';
        hideFiltersBtn.textContent = 'Show';
      } else {
        filtersContainer.style.display = 'block';
        hideFiltersBtn.textContent = 'Hide';
      }
    });
  }
  
  // Tab navigation
  if (weekTab && monthTab) {
    weekTab.addEventListener('click', function() {
      weekTab.classList.add('active');
      monthTab.classList.remove('active');
      updateCalendarView('week');
    });
    
    monthTab.addEventListener('click', function() {
      monthTab.classList.add('active');
      weekTab.classList.remove('active');
      updateCalendarView('month');
    });
  }
  
  // Calendar navigation
  let currentDate = new Date();
  let currentView = 'week';
  
  function updateCalendarView(view) {
    currentView = view;
    updateCalendarTitle();
  }
  
  function updateCalendarTitle() {
    if (!calendarTitle) return;
    
    if (currentView === 'week') {
      // Get first and last day of the current week
      const firstDay = new Date(currentDate);
      firstDay.setDate(currentDate.getDate() - currentDate.getDay());
      
      const lastDay = new Date(firstDay);
      lastDay.setDate(firstDay.getDate() + 6);
      
      // Format dates
      const firstDayFormatted = firstDay.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric'
      });
      
      const lastDayFormatted = lastDay.toLocaleDateString('en-US', {
        month: 'long',
        day: 'numeric',
        year: 'numeric'
      });
      
      calendarTitle.textContent = `${firstDayFormatted} - ${lastDayFormatted}`;
    } else {
      // Month view
      calendarTitle.textContent = currentDate.toLocaleDateString('en-US', {
        month: 'long',
        year: 'numeric'
      });
    }
  }
  
  // Initialize calendar title
  updateCalendarTitle();
  
  // Previous period
  if (prevBtn) {
    prevBtn.addEventListener('click', function() {
      if (currentView === 'week') {
        // Go to previous week
        currentDate.setDate(currentDate.getDate() - 7);
      } else {
        // Go to previous month
        currentDate.setMonth(currentDate.getMonth() - 1);
      }
      
      updateCalendarTitle();
    });
  }
  
  // Next period
  if (nextBtn) {
    nextBtn.addEventListener('click', function() {
      if (currentView === 'week') {
        // Go to next week
        currentDate.setDate(currentDate.getDate() + 7);
      } else {
        // Go to next month
        currentDate.setMonth(currentDate.getMonth() + 1);
      }
      
      updateCalendarTitle();
    });
  }
  
  // Today button
  const todayBtn = document.getElementById('today-btn');
  if (todayBtn) {
    todayBtn.addEventListener('click', function() {
      currentDate = new Date();
      updateCalendarTitle();
    });
  }
  
  // Initialize the dropdown as visible and active
  if (sessionTypeSelect) {
    // Ensure the dropdown is properly initialized and visible
    sessionTypeSelect.style.display = 'block';
    sessionTypeSelect.style.opacity = '1';
    
    // Force repaint/reflow to ensure visibility
    setTimeout(() => {
      sessionTypeSelect.classList.add('visible');
      
      // Add a click handler to ensure dropdown opens properly
      sessionTypeSelect.addEventListener('click', function(e) {
        // This helps with some browsers that might have issues with the dropdown
        e.stopPropagation();
        this.focus();
      });
    }, 100);
  }
}); 
document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    setupNavigation();
    
    const user = getUser();
    document.getElementById('welcomeName').textContent = user.name;

    loadDashboardData();

    if (isAdmin()) {
        document.getElementById('adminStats').style.display = 'block';
        loadAdminStats();
    }
});

async function loadDashboardData() {
    await Promise.all([
        loadUserStats(),
        loadRecentAnnouncements(),
        loadUpcomingBookings(),
        loadMyComplaints()
    ]);
}

async function loadUserStats() {
    try {
        const [bookings, resources, complaints, announcements] = await Promise.all([
            api.get('/bookings/my-bookings'),
            api.get('/resources/my-resources'),
            api.get('/complaints/my-complaints'),
            api.get('/announcements')
        ]);

        document.getElementById('myBookingsCount').textContent = bookings.filter(b => b.status === 'confirmed').length;
        document.getElementById('myResourcesCount').textContent = resources.length;
        document.getElementById('myComplaintsCount').textContent = complaints.length;
        document.getElementById('announcementsCount').textContent = announcements.length;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadAdminStats() {
    try {
        const stats = await api.get('/admin/stats');
        
        document.getElementById('totalUsers').textContent = stats.users.total;
        document.getElementById('totalBookings').textContent = stats.bookings.total;
        document.getElementById('pendingComplaints').textContent = stats.complaints.submitted + stats.complaints.inProgress;
        document.getElementById('totalResources').textContent = stats.resources.total;
    } catch (error) {
        console.error('Error loading admin stats:', error);
    }
}

async function loadRecentAnnouncements() {
    const container = document.getElementById('recentAnnouncements');
    try {
        const announcements = await api.get('/announcements');
        const recent = announcements.slice(0, 3);

        if (recent.length === 0) {
            container.innerHTML = '<p class="empty-state">No announcements yet</p>';
            return;
        }

        container.innerHTML = recent.map(a => `
            <div class="announcement-card priority-${a.priority}">
                <h4>${escapeHtml(a.title)}</h4>
                <p>${escapeHtml(a.content.substring(0, 100))}${a.content.length > 100 ? '...' : ''}</p>
                <div class="announcement-meta">
                    <span class="status-badge status-${a.category}">${a.category}</span>
                    <span>${formatDate(a.createdAt)}</span>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load announcements</p>';
    }
}

async function loadUpcomingBookings() {
    const container = document.getElementById('upcomingBookings');
    try {
        const bookings = await api.get('/bookings/my-bookings');
        const upcoming = bookings
            .filter(b => b.status === 'confirmed' && new Date(b.date) >= new Date())
            .slice(0, 3);

        if (upcoming.length === 0) {
            container.innerHTML = '<p class="empty-state">No upcoming bookings</p>';
            return;
        }

        container.innerHTML = upcoming.map(b => `
            <div class="booking-item">
                <div>
                    <strong>${escapeHtml(b.amenity?.name || 'Unknown')}</strong>
                    <p>${new Date(b.date).toLocaleDateString()} | ${b.startTime} - ${b.endTime}</p>
                </div>
                <span class="status-badge status-${b.status}">${b.status}</span>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load bookings</p>';
    }
}

async function loadMyComplaints() {
    const container = document.getElementById('myComplaints');
    try {
        const complaints = await api.get('/complaints/my-complaints');
        const recent = complaints.slice(0, 3);

        if (recent.length === 0) {
            container.innerHTML = '<p class="empty-state">No complaints filed</p>';
            return;
        }

        container.innerHTML = recent.map(c => `
            <div class="complaint-card priority-${c.priority}">
                <div class="complaint-header">
                    <h4>${escapeHtml(c.title)}</h4>
                    <span class="status-badge status-${c.status}">${c.status.replace('_', ' ')}</span>
                </div>
                <p>${escapeHtml(c.description.substring(0, 80))}${c.description.length > 80 ? '...' : ''}</p>
                <span class="time">${formatDate(c.createdAt)}</span>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load complaints</p>';
    }
}

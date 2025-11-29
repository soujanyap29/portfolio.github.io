const api = {
    async request(endpoint, options = {}) {
        const token = getToken();
        const headers = {
            'Content-Type': 'application/json',
            ...(token && { 'Authorization': `Bearer ${token}` }),
            ...options.headers
        };

        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                ...options,
                headers
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || 'Request failed');
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },

    post(endpoint, body) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(body)
        });
    },

    put(endpoint, body) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(body)
        });
    },

    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
};

function checkAuth() {
    if (!isLoggedIn()) {
        window.location.href = '../index.html';
        return false;
    }
    return true;
}

function setupNavigation() {
    const user = getUser();
    const userName = document.getElementById('userName');
    if (userName && user) {
        userName.textContent = user.name;
    }

    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            window.location.href = '../index.html';
        });
    }

    const notificationBell = document.getElementById('notificationBell');
    const notificationPanel = document.getElementById('notificationPanel');
    if (notificationBell && notificationPanel) {
        notificationBell.addEventListener('click', (e) => {
            e.stopPropagation();
            notificationPanel.classList.toggle('active');
            if (notificationPanel.classList.contains('active')) {
                loadNotifications();
            }
        });

        document.addEventListener('click', (e) => {
            if (!notificationPanel.contains(e.target)) {
                notificationPanel.classList.remove('active');
            }
        });
    }

    const markAllRead = document.getElementById('markAllRead');
    if (markAllRead) {
        markAllRead.addEventListener('click', async () => {
            try {
                await api.put('/notifications/read-all');
                loadNotifications();
                updateNotificationCount();
            } catch (error) {
                console.error('Error marking all as read:', error);
            }
        });
    }

    updateNotificationCount();
}

async function updateNotificationCount() {
    try {
        const data = await api.get('/notifications/unread-count');
        const countElement = document.getElementById('notificationCount');
        if (countElement) {
            countElement.textContent = data.count;
            countElement.style.display = data.count > 0 ? 'block' : 'none';
        }
    } catch (error) {
        console.error('Error fetching notification count:', error);
    }
}

async function loadNotifications() {
    const notificationList = document.getElementById('notificationList');
    if (!notificationList) return;

    try {
        const notifications = await api.get('/notifications');
        
        if (notifications.length === 0) {
            notificationList.innerHTML = '<p class="empty-state">No notifications</p>';
            return;
        }

        notificationList.innerHTML = notifications.map(notification => `
            <div class="notification-item ${notification.isRead ? '' : 'unread'}" 
                 onclick="markNotificationRead('${notification._id}')">
                <h4>${escapeHtml(notification.title)}</h4>
                <p>${escapeHtml(notification.message)}</p>
                <span class="time">${formatDate(notification.createdAt)}</span>
            </div>
        `).join('');
    } catch (error) {
        notificationList.innerHTML = '<p class="error-message">Failed to load notifications</p>';
    }
}

async function markNotificationRead(id) {
    try {
        await api.put(`/notifications/${id}/read`);
        updateNotificationCount();
        loadNotifications();
    } catch (error) {
        console.error('Error marking notification as read:', error);
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`;
    
    return date.toLocaleDateString();
}

function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showModal(modalId) {
    document.getElementById(modalId).classList.add('active');
}

function hideModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function setupTabs(tabBtnSelector, tabContentPrefix) {
    const tabBtns = document.querySelectorAll(tabBtnSelector);
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const tabName = btn.dataset.tab;
            document.querySelectorAll(`[id$="Tab"]`).forEach(content => {
                if (content.id === `${tabName}Tab`) {
                    content.classList.add('active');
                } else if (content.id.endsWith('Tab')) {
                    content.classList.remove('active');
                }
            });
        });
    });
}

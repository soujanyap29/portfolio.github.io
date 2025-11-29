document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    setupNavigation();
    
    if (isAdmin()) {
        document.getElementById('adminActions').style.display = 'block';
        document.getElementById('newAnnouncementBtn').addEventListener('click', () => openAnnouncementModal());
    }

    loadAnnouncements();

    document.getElementById('closeAnnouncementModal').addEventListener('click', () => hideModal('announcementModal'));
    document.getElementById('announcementForm').addEventListener('submit', handleAnnouncementSubmit);
    document.getElementById('categoryFilter').addEventListener('change', loadAnnouncements);
});

async function loadAnnouncements() {
    const container = document.getElementById('announcementsList');
    const categoryFilter = document.getElementById('categoryFilter').value;
    
    try {
        let url = '/announcements';
        if (categoryFilter) {
            url += `?category=${categoryFilter}`;
        }
        
        const announcements = await api.get(url);

        if (announcements.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No announcements yet</p></div>';
            return;
        }

        container.innerHTML = announcements.map(announcement => `
            <div class="announcement-card priority-${announcement.priority}">
                <div class="announcement-header" style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h3>${escapeHtml(announcement.title)}</h3>
                        <div style="margin-top: 5px;">
                            <span class="status-badge status-${announcement.category}">${announcement.category}</span>
                            ${announcement.priority === 'urgent' || announcement.priority === 'high' ? 
                                `<span class="status-badge" style="background: #f8d7da; color: #721c24; margin-left: 5px;">${announcement.priority}</span>` : ''}
                        </div>
                    </div>
                    ${isAdmin() ? `
                        <div>
                            <button class="btn btn-secondary btn-small" onclick="editAnnouncement('${announcement._id}')">Edit</button>
                            <button class="btn btn-danger btn-small" onclick="deleteAnnouncement('${announcement._id}')">Delete</button>
                        </div>
                    ` : ''}
                </div>
                <div class="announcement-content" style="margin: 15px 0;">
                    <p>${escapeHtml(announcement.content)}</p>
                </div>
                <div class="announcement-meta">
                    <span>Posted by: ${escapeHtml(announcement.author?.name || 'Admin')}</span>
                    <span>${formatDate(announcement.createdAt)}</span>
                </div>
                ${announcement.expiresAt ? `
                    <div style="margin-top: 10px; font-size: 0.85rem; color: #999;">
                        Expires: ${new Date(announcement.expiresAt).toLocaleDateString()}
                    </div>
                ` : ''}
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load announcements</p>';
    }
}

function openAnnouncementModal(announcement = null) {
    document.getElementById('announcementModalTitle').textContent = announcement ? 'Edit Announcement' : 'New Announcement';
    document.getElementById('announcementId').value = announcement?._id || '';
    document.getElementById('announcementTitle').value = announcement?.title || '';
    document.getElementById('announcementContent').value = announcement?.content || '';
    document.getElementById('announcementCategory').value = announcement?.category || 'general';
    document.getElementById('announcementPriority').value = announcement?.priority || 'medium';
    document.getElementById('announcementExpiry').value = announcement?.expiresAt ? announcement.expiresAt.split('T')[0] : '';
    document.getElementById('announcementError').textContent = '';
    showModal('announcementModal');
}

async function editAnnouncement(id) {
    try {
        const announcement = await api.get(`/announcements/${id}`);
        openAnnouncementModal(announcement);
    } catch (error) {
        alert('Failed to load announcement: ' + error.message);
    }
}

async function handleAnnouncementSubmit(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('announcementError');
    
    const id = document.getElementById('announcementId').value;
    const data = {
        title: document.getElementById('announcementTitle').value,
        content: document.getElementById('announcementContent').value,
        category: document.getElementById('announcementCategory').value,
        priority: document.getElementById('announcementPriority').value,
        expiresAt: document.getElementById('announcementExpiry').value || null
    };

    try {
        if (id) {
            await api.put(`/announcements/${id}`, data);
        } else {
            await api.post('/announcements', data);
        }
        hideModal('announcementModal');
        loadAnnouncements();
        alert(`Announcement ${id ? 'updated' : 'posted'} successfully!`);
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function deleteAnnouncement(id) {
    if (!confirm('Are you sure you want to delete this announcement?')) return;

    try {
        await api.delete(`/announcements/${id}`);
        loadAnnouncements();
        alert('Announcement deleted successfully');
    } catch (error) {
        alert('Failed to delete announcement: ' + error.message);
    }
}

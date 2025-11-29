document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    
    if (!isAdmin()) {
        alert('Access denied. Admin privileges required.');
        window.location.href = 'dashboard.html';
        return;
    }

    setupNavigation();
    setupAdminTabs();
    
    loadAdminStats();
    loadUsers();
    loadAmenities();
    loadAllComplaints();

    document.getElementById('addAmenityBtn').addEventListener('click', () => openAmenityModal());
    document.getElementById('closeAmenityModal').addEventListener('click', () => hideModal('amenityModal'));
    document.getElementById('closeUpdateComplaintModal').addEventListener('click', () => hideModal('updateComplaintModal'));
    document.getElementById('amenityForm').addEventListener('submit', handleAmenitySubmit);
    document.getElementById('updateComplaintForm').addEventListener('submit', handleComplaintStatusUpdate);
    document.getElementById('adminComplaintStatus').addEventListener('change', loadAllComplaints);
});

function setupAdminTabs() {
    const tabBtns = document.querySelectorAll('.admin-tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            document.querySelectorAll('.admin-tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            document.getElementById(`${btn.dataset.tab}Tab`).classList.add('active');
        });
    });
}

async function loadAdminStats() {
    try {
        const stats = await api.get('/admin/stats');
        
        document.getElementById('statTotalUsers').textContent = stats.users.total;
        document.getElementById('statResidents').textContent = stats.users.residents;
        document.getElementById('statAdmins').textContent = stats.users.admins;
        
        document.getElementById('statTotalBookings').textContent = stats.bookings.total;
        document.getElementById('statConfirmedBookings').textContent = stats.bookings.confirmed;
        
        document.getElementById('statPendingComplaints').textContent = stats.complaints.submitted;
        document.getElementById('statInProgressComplaints').textContent = stats.complaints.inProgress;
        
        document.getElementById('statTotalResources').textContent = stats.resources.total;
        document.getElementById('statAvailableResources').textContent = stats.resources.available;
        document.getElementById('statBorrowedResources').textContent = stats.resources.borrowed;
    } catch (error) {
        console.error('Error loading admin stats:', error);
    }
}

async function loadUsers() {
    const tbody = document.getElementById('usersTableBody');
    
    try {
        const users = await api.get('/admin/users');
        
        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6">No users found</td></tr>';
            return;
        }

        tbody.innerHTML = users.map(user => `
            <tr>
                <td>${escapeHtml(user.name)}</td>
                <td>${escapeHtml(user.email)}</td>
                <td>${escapeHtml(user.flatNumber)}</td>
                <td>${escapeHtml(user.phone)}</td>
                <td>
                    <span class="status-badge status-${user.role === 'admin' ? 'approved' : 'available'}">${user.role}</span>
                </td>
                <td>
                    <button class="btn btn-secondary btn-small" onclick="viewUserDetails('${user._id}')">View</button>
                    ${user.role !== 'admin' ? `
                        <button class="btn btn-warning btn-small" onclick="promoteToAdmin('${user._id}')">Make Admin</button>
                    ` : ''}
                </td>
            </tr>
        `).join('');
    } catch (error) {
        tbody.innerHTML = '<tr><td colspan="6" class="error-message">Failed to load users</td></tr>';
    }
}

async function viewUserDetails(userId) {
    try {
        const data = await api.get(`/admin/users/${userId}`);
        alert(`User: ${data.user.name}\nEmail: ${data.user.email}\nFlat: ${data.user.flatNumber}\nBookings: ${data.bookings.length}\nComplaints: ${data.complaints.length}\nResources: ${data.resources.length}`);
    } catch (error) {
        alert('Failed to load user details: ' + error.message);
    }
}

async function promoteToAdmin(userId) {
    if (!confirm('Are you sure you want to promote this user to admin?')) return;

    try {
        await api.put(`/admin/users/${userId}/role`, { role: 'admin' });
        loadUsers();
        loadAdminStats();
        alert('User promoted to admin successfully');
    } catch (error) {
        alert('Failed to promote user: ' + error.message);
    }
}

async function loadAmenities() {
    const container = document.getElementById('amenitiesAdminList');
    
    try {
        const amenities = await api.get('/amenities');
        
        if (amenities.length === 0) {
            container.innerHTML = '<p class="empty-state">No amenities found</p>';
            return;
        }

        container.innerHTML = amenities.map(amenity => `
            <div class="amenity-admin-card">
                <h3>${escapeHtml(amenity.name)}</h3>
                <p>${escapeHtml(amenity.description)}</p>
                <div class="amenity-details">
                    <span>📍 ${escapeHtml(amenity.location || 'Not specified')}</span>
                    <span>🕐 ${amenity.operatingHours.start} - ${amenity.operatingHours.end}</span>
                    <span>👥 Capacity: ${amenity.capacity}</span>
                    <span class="status-badge status-${amenity.isActive ? 'available' : 'unavailable'}">
                        ${amenity.isActive ? 'Active' : 'Inactive'}
                    </span>
                </div>
                <div class="card-actions">
                    <button class="btn btn-secondary btn-small" onclick="editAmenity('${amenity._id}')">Edit</button>
                    ${amenity.isActive ? `
                        <button class="btn btn-danger btn-small" onclick="deactivateAmenity('${amenity._id}')">Deactivate</button>
                    ` : `
                        <button class="btn btn-success btn-small" onclick="activateAmenity('${amenity._id}')">Activate</button>
                    `}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load amenities</p>';
    }
}

function openAmenityModal(amenity = null) {
    document.getElementById('amenityModalTitle').textContent = amenity ? 'Edit Amenity' : 'Add Amenity';
    document.getElementById('amenityId').value = amenity?._id || '';
    document.getElementById('amenityName').value = amenity?.name || '';
    document.getElementById('amenityDescription').value = amenity?.description || '';
    document.getElementById('amenityLocation').value = amenity?.location || '';
    document.getElementById('amenityStartTime').value = amenity?.operatingHours?.start || '06:00';
    document.getElementById('amenityEndTime').value = amenity?.operatingHours?.end || '22:00';
    document.getElementById('amenityCapacity').value = amenity?.capacity || 1;
    document.getElementById('amenityError').textContent = '';
    showModal('amenityModal');
}

async function editAmenity(id) {
    try {
        const amenity = await api.get(`/amenities/${id}`);
        openAmenityModal(amenity);
    } catch (error) {
        alert('Failed to load amenity: ' + error.message);
    }
}

async function handleAmenitySubmit(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('amenityError');
    
    const id = document.getElementById('amenityId').value;
    const data = {
        name: document.getElementById('amenityName').value,
        description: document.getElementById('amenityDescription').value,
        location: document.getElementById('amenityLocation').value,
        operatingHours: {
            start: document.getElementById('amenityStartTime').value,
            end: document.getElementById('amenityEndTime').value
        },
        capacity: parseInt(document.getElementById('amenityCapacity').value)
    };

    try {
        if (id) {
            await api.put(`/amenities/${id}`, data);
        } else {
            await api.post('/amenities', data);
        }
        hideModal('amenityModal');
        loadAmenities();
        alert(`Amenity ${id ? 'updated' : 'added'} successfully!`);
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function deactivateAmenity(id) {
    if (!confirm('Are you sure you want to deactivate this amenity?')) return;

    try {
        await api.delete(`/amenities/${id}`);
        loadAmenities();
        alert('Amenity deactivated');
    } catch (error) {
        alert('Failed to deactivate amenity: ' + error.message);
    }
}

async function activateAmenity(id) {
    try {
        await api.put(`/amenities/${id}`, { isActive: true });
        loadAmenities();
        alert('Amenity activated');
    } catch (error) {
        alert('Failed to activate amenity: ' + error.message);
    }
}

async function loadAllComplaints() {
    const container = document.getElementById('allComplaintsList');
    const statusFilter = document.getElementById('adminComplaintStatus').value;
    
    try {
        let url = '/complaints';
        if (statusFilter) {
            url += `?status=${statusFilter}`;
        }
        
        const complaints = await api.get(url);

        if (complaints.length === 0) {
            container.innerHTML = '<p class="empty-state">No complaints found</p>';
            return;
        }

        container.innerHTML = complaints.map(complaint => `
            <div class="complaint-admin-card priority-${complaint.priority}">
                <div class="complaint-admin-header">
                    <div>
                        <h4>${escapeHtml(complaint.title)}</h4>
                        <span style="color: var(--secondary-color); font-size: 0.9rem;">
                            by ${escapeHtml(complaint.user?.name || 'Unknown')} (${escapeHtml(complaint.user?.flatNumber || '')})
                        </span>
                    </div>
                    <span class="status-badge status-${complaint.status}">${complaint.status.replace('_', ' ')}</span>
                </div>
                <div class="complaint-admin-body">
                    <p>${escapeHtml(complaint.description)}</p>
                    <div class="complaint-admin-meta">
                        <span>Category: ${complaint.category}</span>
                        <span>Priority: ${complaint.priority}</span>
                        <span>${formatDate(complaint.createdAt)}</span>
                    </div>
                </div>
                ${complaint.adminNotes ? `
                    <div class="admin-notes">
                        <strong>Admin Notes:</strong> ${escapeHtml(complaint.adminNotes)}
                    </div>
                ` : ''}
                <div class="complaint-admin-footer">
                    <span>
                        ${complaint.user?.email ? `📧 ${escapeHtml(complaint.user.email)}` : ''}
                    </span>
                    ${complaint.status !== 'resolved' ? `
                        <button class="btn btn-primary btn-small" onclick="openUpdateComplaintModal('${complaint._id}', '${escapeHtml(complaint.title)}', '${complaint.status}')">
                            Update Status
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load complaints</p>';
    }
}

function openUpdateComplaintModal(id, title, currentStatus) {
    document.getElementById('updateComplaintId').value = id;
    document.getElementById('complaintDetails').innerHTML = `
        <h4>${escapeHtml(title)}</h4>
        <p>Current Status: ${currentStatus.replace('_', ' ')}</p>
    `;
    
    const statusSelect = document.getElementById('updateComplaintStatus');
    statusSelect.innerHTML = '';
    
    if (currentStatus === 'submitted') {
        statusSelect.innerHTML = `
            <option value="in_progress">In Progress</option>
            <option value="resolved">Resolved</option>
        `;
    } else if (currentStatus === 'in_progress') {
        statusSelect.innerHTML = `
            <option value="resolved">Resolved</option>
        `;
    }
    
    document.getElementById('adminNotes').value = '';
    document.getElementById('updateComplaintError').textContent = '';
    showModal('updateComplaintModal');
}

async function handleComplaintStatusUpdate(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('updateComplaintError');
    
    const id = document.getElementById('updateComplaintId').value;
    const status = document.getElementById('updateComplaintStatus').value;
    const adminNotes = document.getElementById('adminNotes').value;

    try {
        await api.put(`/complaints/${id}/status`, { status, adminNotes });
        hideModal('updateComplaintModal');
        loadAllComplaints();
        loadAdminStats();
        alert('Complaint status updated successfully');
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

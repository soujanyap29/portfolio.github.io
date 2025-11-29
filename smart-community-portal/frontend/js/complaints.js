document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    setupNavigation();
    
    loadComplaints();

    document.getElementById('newComplaintBtn').addEventListener('click', () => openComplaintModal());
    document.getElementById('closeComplaintModal').addEventListener('click', () => hideModal('complaintModal'));
    document.getElementById('closeViewComplaintModal').addEventListener('click', () => hideModal('viewComplaintModal'));
    document.getElementById('complaintForm').addEventListener('submit', handleComplaintSubmit);
    document.getElementById('complaintStatusFilter').addEventListener('change', loadComplaints);
    document.getElementById('complaintCategoryFilter').addEventListener('change', loadComplaints);
});

async function loadComplaints() {
    const container = document.getElementById('complaintsList');
    const statusFilter = document.getElementById('complaintStatusFilter').value;
    const categoryFilter = document.getElementById('complaintCategoryFilter').value;
    
    try {
        let url = '/complaints/my-complaints';
        const complaints = await api.get(url);
        
        let filtered = complaints;
        if (statusFilter) {
            filtered = filtered.filter(c => c.status === statusFilter);
        }
        if (categoryFilter) {
            filtered = filtered.filter(c => c.category === categoryFilter);
        }

        if (filtered.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No complaints found</p><button class="btn btn-primary" onclick="openComplaintModal()">Submit a Complaint</button></div>';
            return;
        }

        container.innerHTML = filtered.map(complaint => `
            <div class="complaint-card priority-${complaint.priority}">
                <div class="complaint-header">
                    <h4>${escapeHtml(complaint.title)}</h4>
                    <span class="status-badge status-${complaint.status}">${complaint.status.replace('_', ' ')}</span>
                </div>
                <div class="complaint-body">
                    <p>${escapeHtml(complaint.description)}</p>
                </div>
                <div class="complaint-footer">
                    <div>
                        <span class="status-badge">${complaint.category}</span>
                        <span style="margin-left: 10px;">Priority: ${complaint.priority}</span>
                    </div>
                    <span>${formatDate(complaint.createdAt)}</span>
                </div>
                ${complaint.adminNotes ? `
                    <div class="admin-notes">
                        <strong>Admin Notes:</strong> ${escapeHtml(complaint.adminNotes)}
                    </div>
                ` : ''}
                <div class="card-actions" style="margin-top: 15px;">
                    <button class="btn btn-secondary btn-small" onclick="viewComplaint('${complaint._id}')">View Details</button>
                    ${complaint.status === 'submitted' ? `
                        <button class="btn btn-primary btn-small" onclick="editComplaint('${complaint._id}')">Edit</button>
                        <button class="btn btn-danger btn-small" onclick="deleteComplaint('${complaint._id}')">Delete</button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load complaints</p>';
    }
}

function openComplaintModal(complaint = null) {
    document.getElementById('complaintModalTitle').textContent = complaint ? 'Edit Complaint' : 'New Complaint';
    document.getElementById('complaintId').value = complaint?._id || '';
    document.getElementById('complaintTitle').value = complaint?.title || '';
    document.getElementById('complaintDescription').value = complaint?.description || '';
    document.getElementById('complaintCategory').value = complaint?.category || 'maintenance';
    document.getElementById('complaintPriority').value = complaint?.priority || 'medium';
    document.getElementById('complaintImage').value = complaint?.imageUrl || '';
    document.getElementById('complaintError').textContent = '';
    showModal('complaintModal');
}

async function editComplaint(id) {
    try {
        const complaint = await api.get(`/complaints/${id}`);
        openComplaintModal(complaint);
    } catch (error) {
        alert('Failed to load complaint: ' + error.message);
    }
}

async function viewComplaint(id) {
    try {
        const complaint = await api.get(`/complaints/${id}`);
        const content = document.getElementById('viewComplaintContent');
        
        content.innerHTML = `
            <div class="complaint-detail">
                <div class="detail-header">
                    <h3>${escapeHtml(complaint.title)}</h3>
                    <span class="status-badge status-${complaint.status}">${complaint.status.replace('_', ' ')}</span>
                </div>
                <div class="detail-meta">
                    <span>Category: ${complaint.category}</span>
                    <span>Priority: ${complaint.priority}</span>
                    <span>Created: ${formatDateTime(complaint.createdAt)}</span>
                </div>
                <div class="detail-body">
                    <h4>Description</h4>
                    <p>${escapeHtml(complaint.description)}</p>
                </div>
                ${complaint.imageUrl ? `
                    <div class="detail-image">
                        <h4>Attached Image</h4>
                        <img src="${escapeHtml(complaint.imageUrl)}" alt="Evidence" style="max-width: 100%; border-radius: 8px;">
                    </div>
                ` : ''}
                ${complaint.adminNotes ? `
                    <div class="admin-notes">
                        <h4>Admin Response</h4>
                        <p>${escapeHtml(complaint.adminNotes)}</p>
                    </div>
                ` : ''}
                ${complaint.resolvedAt ? `
                    <div class="resolved-info">
                        <p>Resolved on: ${formatDateTime(complaint.resolvedAt)}</p>
                    </div>
                ` : ''}
            </div>
        `;
        
        showModal('viewComplaintModal');
    } catch (error) {
        alert('Failed to load complaint details: ' + error.message);
    }
}

async function handleComplaintSubmit(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('complaintError');
    
    const id = document.getElementById('complaintId').value;
    const data = {
        title: document.getElementById('complaintTitle').value,
        description: document.getElementById('complaintDescription').value,
        category: document.getElementById('complaintCategory').value,
        priority: document.getElementById('complaintPriority').value,
        imageUrl: document.getElementById('complaintImage').value
    };

    try {
        if (id) {
            await api.put(`/complaints/${id}`, data);
        } else {
            await api.post('/complaints', data);
        }
        hideModal('complaintModal');
        loadComplaints();
        alert(`Complaint ${id ? 'updated' : 'submitted'} successfully!`);
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function deleteComplaint(id) {
    if (!confirm('Are you sure you want to delete this complaint?')) return;

    try {
        await api.delete(`/complaints/${id}`);
        loadComplaints();
        alert('Complaint deleted successfully');
    } catch (error) {
        alert('Failed to delete complaint: ' + error.message);
    }
}

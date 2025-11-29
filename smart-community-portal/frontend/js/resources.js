document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    setupNavigation();
    setupTabs('.tabs-container .tab-btn', '');
    setupSubTabs();
    
    loadResources();
    loadMyResources();
    loadBorrowRequests();

    document.getElementById('categoryFilter').addEventListener('change', loadResources);
    document.getElementById('statusFilter').addEventListener('change', loadResources);
    document.getElementById('addResourceBtn').addEventListener('click', () => openResourceModal());
    document.getElementById('closeResourceModal').addEventListener('click', () => hideModal('resourceModal'));
    document.getElementById('closeBorrowModal').addEventListener('click', () => hideModal('borrowModal'));
    document.getElementById('resourceForm').addEventListener('submit', handleResourceSubmit);
    document.getElementById('borrowForm').addEventListener('submit', handleBorrowRequest);
});

function setupSubTabs() {
    const subTabBtns = document.querySelectorAll('.tab-btn-inner');
    subTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            subTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            document.querySelectorAll('.subtab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            document.getElementById(`${btn.dataset.subtab}Requests`).classList.add('active');
        });
    });
}

async function loadResources() {
    const container = document.getElementById('resourcesList');
    const category = document.getElementById('categoryFilter').value;
    const status = document.getElementById('statusFilter').value;
    
    try {
        let url = '/resources?';
        if (category) url += `category=${category}&`;
        if (status) url += `status=${status}`;
        
        const resources = await api.get(url);
        const user = getUser();
        
        if (resources.length === 0) {
            container.innerHTML = '<p class="empty-state">No resources found</p>';
            return;
        }

        container.innerHTML = resources.map(resource => `
            <div class="resource-card">
                ${resource.imageUrl ? `<img src="${escapeHtml(resource.imageUrl)}" alt="${escapeHtml(resource.name)}" style="width:100%; height:150px; object-fit:cover; border-radius:8px; margin-bottom:10px;">` : ''}
                <h3>${escapeHtml(resource.name)}</h3>
                <p>${escapeHtml(resource.description)}</p>
                <div class="resource-info">
                    <span>📁 ${resource.category}</span>
                    <span>👤 ${escapeHtml(resource.owner?.name || 'Unknown')} (${escapeHtml(resource.owner?.flatNumber || '')})</span>
                    <span class="status-badge status-${resource.status}">${resource.status}</span>
                </div>
                <div class="card-actions">
                    ${resource.status === 'available' && resource.owner?._id !== user._id ? `
                        <button class="btn btn-primary" onclick="openBorrowModal('${resource._id}', '${escapeHtml(resource.name)}', '${escapeHtml(resource.owner?.name || '')}')">
                            Request to Borrow
                        </button>
                    ` : resource.owner?._id === user._id ? `
                        <span style="color: var(--secondary-color);">Your resource</span>
                    ` : `
                        <span style="color: var(--secondary-color);">Currently unavailable</span>
                    `}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load resources</p>';
    }
}

async function loadMyResources() {
    const container = document.getElementById('myResourcesList');
    
    try {
        const resources = await api.get('/resources/my-resources');
        
        if (resources.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>You haven\'t shared any resources yet</p></div>';
            return;
        }

        container.innerHTML = resources.map(resource => `
            <div class="resource-card">
                <h3>${escapeHtml(resource.name)}</h3>
                <p>${escapeHtml(resource.description)}</p>
                <div class="resource-info">
                    <span>📁 ${resource.category}</span>
                    <span class="status-badge status-${resource.status}">${resource.status}</span>
                    ${resource.currentBorrower ? `<span>Borrowed by: ${escapeHtml(resource.currentBorrower.name)}</span>` : ''}
                </div>
                <div class="card-actions">
                    <button class="btn btn-secondary btn-small" onclick="editResource('${resource._id}')">Edit</button>
                    ${resource.status !== 'borrowed' ? `
                        <button class="btn btn-danger btn-small" onclick="deleteResource('${resource._id}')">Delete</button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load your resources</p>';
    }
}

async function loadBorrowRequests() {
    await Promise.all([loadReceivedRequests(), loadSentRequests()]);
}

async function loadReceivedRequests() {
    const container = document.getElementById('receivedRequestsList');
    
    try {
        const requests = await api.get('/borrow-requests?type=received');
        
        if (requests.length === 0) {
            container.innerHTML = '<p class="empty-state">No borrow requests received</p>';
            return;
        }

        container.innerHTML = requests.map(req => `
            <div class="request-card">
                <div class="request-info">
                    <h4>${escapeHtml(req.resource?.name || 'Unknown Resource')}</h4>
                    <p>Requested by: ${escapeHtml(req.borrower?.name || 'Unknown')} (${escapeHtml(req.borrower?.flatNumber || '')})</p>
                    ${req.message ? `<p>"${escapeHtml(req.message)}"</p>` : ''}
                    <small>${formatDate(req.createdAt)}</small>
                </div>
                <div class="request-actions">
                    <span class="status-badge status-${req.status}">${req.status}</span>
                    ${req.status === 'pending' ? `
                        <button class="btn btn-success btn-small" onclick="approveRequest('${req._id}')">Approve</button>
                        <button class="btn btn-danger btn-small" onclick="rejectRequest('${req._id}')">Reject</button>
                    ` : req.status === 'approved' ? `
                        <button class="btn btn-primary btn-small" onclick="markReturned('${req._id}')">Mark Returned</button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load requests</p>';
    }
}

async function loadSentRequests() {
    const container = document.getElementById('sentRequestsList');
    
    try {
        const requests = await api.get('/borrow-requests?type=sent');
        
        if (requests.length === 0) {
            container.innerHTML = '<p class="empty-state">You haven\'t sent any borrow requests</p>';
            return;
        }

        container.innerHTML = requests.map(req => `
            <div class="request-card">
                <div class="request-info">
                    <h4>${escapeHtml(req.resource?.name || 'Unknown Resource')}</h4>
                    <p>Owner: ${escapeHtml(req.owner?.name || 'Unknown')} (${escapeHtml(req.owner?.flatNumber || '')})</p>
                    <small>${formatDate(req.createdAt)}</small>
                </div>
                <div class="request-actions">
                    <span class="status-badge status-${req.status}">${req.status}</span>
                    ${req.status === 'approved' ? `
                        <button class="btn btn-primary btn-small" onclick="markReturned('${req._id}')">Return Item</button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load requests</p>';
    }
}

function openResourceModal(resource = null) {
    document.getElementById('resourceModalTitle').textContent = resource ? 'Edit Resource' : 'Add Resource';
    document.getElementById('resourceId').value = resource?._id || '';
    document.getElementById('resourceName').value = resource?.name || '';
    document.getElementById('resourceDescription').value = resource?.description || '';
    document.getElementById('resourceCategory').value = resource?.category || 'other';
    document.getElementById('resourceImage').value = resource?.imageUrl || '';
    document.getElementById('resourceError').textContent = '';
    showModal('resourceModal');
}

async function editResource(id) {
    try {
        const resource = await api.get(`/resources/${id}`);
        openResourceModal(resource);
    } catch (error) {
        alert('Failed to load resource: ' + error.message);
    }
}

async function handleResourceSubmit(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('resourceError');
    
    const id = document.getElementById('resourceId').value;
    const data = {
        name: document.getElementById('resourceName').value,
        description: document.getElementById('resourceDescription').value,
        category: document.getElementById('resourceCategory').value,
        imageUrl: document.getElementById('resourceImage').value
    };

    try {
        if (id) {
            await api.put(`/resources/${id}`, data);
        } else {
            await api.post('/resources', data);
        }
        hideModal('resourceModal');
        loadMyResources();
        loadResources();
        alert(`Resource ${id ? 'updated' : 'added'} successfully!`);
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function deleteResource(id) {
    if (!confirm('Are you sure you want to delete this resource?')) return;

    try {
        await api.delete(`/resources/${id}`);
        loadMyResources();
        loadResources();
        alert('Resource deleted successfully');
    } catch (error) {
        alert('Failed to delete resource: ' + error.message);
    }
}

function openBorrowModal(resourceId, resourceName, ownerName) {
    document.getElementById('borrowResourceId').value = resourceId;
    document.getElementById('borrowResourceInfo').innerHTML = `
        <h4>${escapeHtml(resourceName)}</h4>
        <p>Owner: ${escapeHtml(ownerName)}</p>
    `;
    
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('borrowDate').min = today;
    document.getElementById('borrowDate').value = today;
    document.getElementById('returnDate').min = today;
    document.getElementById('returnDate').value = '';
    document.getElementById('borrowMessage').value = '';
    document.getElementById('borrowError').textContent = '';
    
    showModal('borrowModal');
}

async function handleBorrowRequest(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('borrowError');
    
    const resourceId = document.getElementById('borrowResourceId').value;
    const message = document.getElementById('borrowMessage').value;
    const borrowDate = document.getElementById('borrowDate').value;
    const returnDate = document.getElementById('returnDate').value;

    if (returnDate && returnDate < borrowDate) {
        errorDiv.textContent = 'Return date must be after borrow date';
        return;
    }

    try {
        await api.post('/borrow-requests', { resourceId, message, borrowDate, returnDate });
        hideModal('borrowModal');
        loadResources();
        loadBorrowRequests();
        alert('Borrow request sent successfully!');
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function approveRequest(id) {
    try {
        await api.put(`/borrow-requests/${id}/approve`);
        loadBorrowRequests();
        loadMyResources();
        alert('Request approved successfully!');
    } catch (error) {
        alert('Failed to approve request: ' + error.message);
    }
}

async function rejectRequest(id) {
    if (!confirm('Are you sure you want to reject this request?')) return;

    try {
        await api.put(`/borrow-requests/${id}/reject`);
        loadBorrowRequests();
        alert('Request rejected');
    } catch (error) {
        alert('Failed to reject request: ' + error.message);
    }
}

async function markReturned(id) {
    try {
        await api.put(`/borrow-requests/${id}/return`);
        loadBorrowRequests();
        loadMyResources();
        loadResources();
        alert('Item marked as returned!');
    } catch (error) {
        alert('Failed to mark as returned: ' + error.message);
    }
}

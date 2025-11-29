document.addEventListener('DOMContentLoaded', () => {
    if (!checkAuth()) return;
    setupNavigation();
    setupTabs('.tabs-container .tab-btn', '');
    
    loadAmenities();
    loadMyBookings();

    document.getElementById('closeBookingModal').addEventListener('click', () => hideModal('bookingModal'));
    document.getElementById('bookingForm').addEventListener('submit', handleBooking);
    document.getElementById('bookingStatusFilter').addEventListener('change', loadMyBookings);
    document.getElementById('bookingDate').addEventListener('change', loadExistingBookings);
});

async function loadAmenities() {
    const container = document.getElementById('amenitiesList');
    try {
        const amenities = await api.get('/amenities');
        
        if (amenities.length === 0) {
            container.innerHTML = '<p class="empty-state">No amenities available</p>';
            return;
        }

        container.innerHTML = amenities.map(amenity => `
            <div class="amenity-card">
                <h3>${escapeHtml(amenity.name)}</h3>
                <p>${escapeHtml(amenity.description)}</p>
                <div class="amenity-info">
                    <span>📍 ${escapeHtml(amenity.location || 'Not specified')}</span>
                    <span>🕐 ${amenity.operatingHours.start} - ${amenity.operatingHours.end}</span>
                    <span>👥 Capacity: ${amenity.capacity}</span>
                </div>
                <div class="card-actions">
                    <button class="btn btn-primary" onclick="openBookingModal('${amenity._id}', '${escapeHtml(amenity.name)}', '${amenity.operatingHours.start}', '${amenity.operatingHours.end}')">
                        Book Now
                    </button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load amenities</p>';
    }
}

async function loadMyBookings() {
    const container = document.getElementById('myBookingsList');
    const statusFilter = document.getElementById('bookingStatusFilter').value;
    
    try {
        let url = '/bookings/my-bookings';
        const bookings = await api.get(url);
        
        let filtered = bookings;
        if (statusFilter) {
            filtered = bookings.filter(b => b.status === statusFilter);
        }

        if (filtered.length === 0) {
            container.innerHTML = '<p class="empty-state">No bookings found</p>';
            return;
        }

        container.innerHTML = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Amenity</th>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${filtered.map(booking => `
                        <tr>
                            <td>${escapeHtml(booking.amenity?.name || 'Unknown')}</td>
                            <td>${new Date(booking.date).toLocaleDateString()}</td>
                            <td>${booking.startTime} - ${booking.endTime}</td>
                            <td><span class="status-badge status-${booking.status}">${booking.status}</span></td>
                            <td>
                                ${booking.status === 'confirmed' ? `
                                    <button class="btn btn-danger btn-small" onclick="cancelBooking('${booking._id}')">
                                        Cancel
                                    </button>
                                ` : '-'}
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        container.innerHTML = '<p class="error-message">Failed to load bookings</p>';
    }
}

function openBookingModal(amenityId, amenityName, startHour, endHour) {
    document.getElementById('bookingAmenityId').value = amenityId;
    document.getElementById('selectedAmenityInfo').innerHTML = `
        <h4>Booking: ${escapeHtml(amenityName)}</h4>
        <p>Operating Hours: ${startHour} - ${endHour}</p>
    `;
    
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('bookingDate').min = today;
    document.getElementById('bookingDate').value = today;
    document.getElementById('bookingStartTime').value = startHour;
    document.getElementById('bookingEndTime').value = '';
    document.getElementById('bookingNotes').value = '';
    document.getElementById('bookingError').textContent = '';
    document.getElementById('existingBookings').innerHTML = '';
    
    showModal('bookingModal');
    loadExistingBookings();
}

async function loadExistingBookings() {
    const amenityId = document.getElementById('bookingAmenityId').value;
    const date = document.getElementById('bookingDate').value;
    const container = document.getElementById('existingBookings');
    
    if (!amenityId || !date) return;

    try {
        const bookings = await api.get(`/bookings/amenity/${amenityId}?date=${date}`);
        
        if (bookings.length === 0) {
            container.innerHTML = '<p style="color: green;">✓ No existing bookings for this date</p>';
            return;
        }

        container.innerHTML = `
            <h4>Existing Bookings on ${new Date(date).toLocaleDateString()}:</h4>
            <ul>
                ${bookings.map(b => `
                    <li>${b.startTime} - ${b.endTime} (${escapeHtml(b.user?.name || 'Unknown')})</li>
                `).join('')}
            </ul>
        `;
    } catch (error) {
        container.innerHTML = '';
    }
}

async function handleBooking(e) {
    e.preventDefault();
    const errorDiv = document.getElementById('bookingError');
    
    const amenityId = document.getElementById('bookingAmenityId').value;
    const date = document.getElementById('bookingDate').value;
    const startTime = document.getElementById('bookingStartTime').value;
    const endTime = document.getElementById('bookingEndTime').value;
    const notes = document.getElementById('bookingNotes').value;

    if (startTime >= endTime) {
        errorDiv.textContent = 'End time must be after start time';
        return;
    }

    try {
        await api.post('/bookings', { amenityId, date, startTime, endTime, notes });
        hideModal('bookingModal');
        loadMyBookings();
        alert('Booking confirmed successfully!');
    } catch (error) {
        errorDiv.textContent = error.message;
    }
}

async function cancelBooking(bookingId) {
    if (!confirm('Are you sure you want to cancel this booking?')) return;

    try {
        await api.delete(`/bookings/${bookingId}`);
        loadMyBookings();
        alert('Booking cancelled successfully');
    } catch (error) {
        alert('Failed to cancel booking: ' + error.message);
    }
}

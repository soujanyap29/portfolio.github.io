require('dotenv').config();
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');

const authRoutes = require('./routes/auth');
const amenityRoutes = require('./routes/amenities');
const bookingRoutes = require('./routes/bookings');
const resourceRoutes = require('./routes/resources');
const borrowRequestRoutes = require('./routes/borrowRequests');
const complaintRoutes = require('./routes/complaints');
const announcementRoutes = require('./routes/announcements');
const notificationRoutes = require('./routes/notifications');
const adminRoutes = require('./routes/admin');

const app = express();

connectDB();

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.json({ 
        message: 'Smart Community Resource Sharing Portal API',
        version: '1.0.0',
        endpoints: {
            auth: '/api/auth',
            amenities: '/api/amenities',
            bookings: '/api/bookings',
            resources: '/api/resources',
            borrowRequests: '/api/borrow-requests',
            complaints: '/api/complaints',
            announcements: '/api/announcements',
            notifications: '/api/notifications',
            admin: '/api/admin'
        }
    });
});

app.use('/api/auth', authRoutes);
app.use('/api/amenities', amenityRoutes);
app.use('/api/bookings', bookingRoutes);
app.use('/api/resources', resourceRoutes);
app.use('/api/borrow-requests', borrowRequestRoutes);
app.use('/api/complaints', complaintRoutes);
app.use('/api/announcements', announcementRoutes);
app.use('/api/notifications', notificationRoutes);
app.use('/api/admin', adminRoutes);

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ message: 'Something went wrong!' });
});

app.use((req, res) => {
    res.status(404).json({ message: 'Route not found' });
});

const PORT = process.env.PORT || 4000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

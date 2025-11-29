# Smart Community Resource Sharing Portal

A complete full-stack web application for community management, built with Node.js, MongoDB, and vanilla JavaScript frontend. This portal allows residents and administrators to manage community amenities, share resources, handle complaints, and stay updated with announcements.

## 🌟 Features

### For Residents
- **User Authentication**: Register and login with secure password protection
- **Amenity Booking**: Book community amenities like Gym, Clubhouse, Swimming Pool, Tennis Court, and Guest Rooms
- **Resource Sharing**: Share personal items with neighbors and borrow from others
- **Complaint Management**: Submit and track complaints related to maintenance, security, facilities, etc.
- **Announcements**: Stay updated with community notices and events
- **Notifications**: Receive real-time notifications for booking updates, complaint status changes, and borrow requests

### For Administrators
- **User Management**: View all residents and their activities
- **Amenity Management**: Add, edit, activate/deactivate community amenities
- **Complaint Processing**: Update complaint status (Submitted → In Progress → Resolved) with admin notes
- **Announcement Publishing**: Post community announcements with categories and priority levels
- **Dashboard Statistics**: Monitor community-wide metrics

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Node.js with Express.js
- **Database**: MongoDB
- **Containerization**: Docker & Docker Compose

## 📁 Project Structure

```
smart-community-portal/
├── backend/
│   ├── config/
│   │   └── db.js              # MongoDB connection
│   ├── middleware/
│   │   └── auth.js            # JWT authentication middleware
│   ├── models/
│   │   ├── User.js            # User model
│   │   ├── Amenity.js         # Amenity model
│   │   ├── Booking.js         # Booking model
│   │   ├── Resource.js        # Resource model
│   │   ├── BorrowRequest.js   # Borrow request model
│   │   ├── Complaint.js       # Complaint model
│   │   ├── Announcement.js    # Announcement model
│   │   └── Notification.js    # Notification model
│   ├── routes/
│   │   ├── auth.js            # Authentication routes
│   │   ├── amenities.js       # Amenity routes
│   │   ├── bookings.js        # Booking routes
│   │   ├── resources.js       # Resource routes
│   │   ├── borrowRequests.js  # Borrow request routes
│   │   ├── complaints.js      # Complaint routes
│   │   ├── announcements.js   # Announcement routes
│   │   ├── notifications.js   # Notification routes
│   │   └── admin.js           # Admin routes
│   ├── Dockerfile
│   ├── package.json
│   ├── server.js              # Main server file
│   └── seed.js                # Database seeding script
├── frontend/
│   ├── css/
│   │   ├── styles.css         # Main styles
│   │   ├── dashboard.css      # Dashboard styles
│   │   └── admin.css          # Admin panel styles
│   ├── js/
│   │   ├── config.js          # API configuration
│   │   ├── api.js             # API helper functions
│   │   ├── auth.js            # Authentication logic
│   │   ├── dashboard.js       # Dashboard logic
│   │   ├── amenities.js       # Amenities & bookings logic
│   │   ├── resources.js       # Resource sharing logic
│   │   ├── complaints.js      # Complaints logic
│   │   ├── announcements.js   # Announcements logic
│   │   └── admin.js           # Admin panel logic
│   ├── pages/
│   │   ├── dashboard.html     # User dashboard
│   │   ├── amenities.html     # Amenities & bookings page
│   │   ├── resources.html     # Resource sharing page
│   │   ├── complaints.html    # Complaints page
│   │   ├── announcements.html # Announcements page
│   │   └── admin.html         # Admin panel
│   └── index.html             # Login/Register page
└── docker-compose.yml         # Docker orchestration
```

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose installed
- Git

### Installation & Running

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd smart-community-portal
   ```

2. **Start the application with Docker**
   ```bash
   docker compose up --build
   ```

3. **Access the application**
   - Open `frontend/index.html` in your web browser
   - Backend API will be running at `http://localhost:4000`
   - MongoDB will be available at `localhost:27017`

4. **Seed the database (Optional but recommended)**
   ```bash
   docker exec -it <backend-container-id> npm run seed
   ```

### Default Login Credentials (After Seeding)

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@community.com | Admin@123 |
| Resident | john@resident.com | Password@123 |
| Resident | jane@resident.com | Password@123 |
| Resident | mike@resident.com | Password@123 |

## 📊 MongoDB Collections

| Collection | Description |
|------------|-------------|
| users | User accounts with roles |
| amenities | Community amenities details |
| bookings | Amenity reservations |
| resources | Shared community resources |
| borrowrequests | Resource borrowing workflow |
| complaints | Issue reports and resolutions |
| announcements | Community notices |
| notifications | User notifications |

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update profile

### Amenities
- `GET /api/amenities` - List all amenities
- `GET /api/amenities/:id` - Get amenity details
- `POST /api/amenities` - Create amenity (Admin)
- `PUT /api/amenities/:id` - Update amenity (Admin)
- `DELETE /api/amenities/:id` - Deactivate amenity (Admin)

### Bookings
- `GET /api/bookings` - List bookings
- `GET /api/bookings/my-bookings` - Get user's bookings
- `POST /api/bookings` - Create booking
- `PUT /api/bookings/:id` - Update booking
- `DELETE /api/bookings/:id` - Cancel booking

### Resources
- `GET /api/resources` - List all resources
- `GET /api/resources/my-resources` - Get user's resources
- `POST /api/resources` - Share a resource
- `PUT /api/resources/:id` - Update resource
- `DELETE /api/resources/:id` - Delete resource

### Borrow Requests
- `GET /api/borrow-requests` - List requests
- `POST /api/borrow-requests` - Send borrow request
- `PUT /api/borrow-requests/:id/approve` - Approve request
- `PUT /api/borrow-requests/:id/reject` - Reject request
- `PUT /api/borrow-requests/:id/return` - Mark as returned

### Complaints
- `GET /api/complaints` - List complaints
- `POST /api/complaints` - Submit complaint
- `PUT /api/complaints/:id` - Update complaint
- `PUT /api/complaints/:id/status` - Update status (Admin)
- `DELETE /api/complaints/:id` - Delete complaint

### Announcements
- `GET /api/announcements` - List announcements
- `POST /api/announcements` - Create announcement (Admin)
- `PUT /api/announcements/:id` - Update announcement (Admin)
- `DELETE /api/announcements/:id` - Delete announcement (Admin)

### Notifications
- `GET /api/notifications` - Get user notifications
- `GET /api/notifications/unread-count` - Get unread count
- `PUT /api/notifications/:id/read` - Mark as read
- `PUT /api/notifications/read-all` - Mark all as read

### Admin
- `GET /api/admin/users` - List all users
- `GET /api/admin/users/:id` - Get user details
- `PUT /api/admin/users/:id/role` - Update user role
- `GET /api/admin/stats` - Get dashboard statistics

## 📱 Using MongoDB Compass

1. Open MongoDB Compass
2. Click "New Connection"
3. Enter: `mongodb://localhost:27017`
4. Connect to the server
5. Select database: `smart_community`
6. Browse and manage collections

## ⚙️ Booking Conflict Logic

The system prevents double bookings using the following rule:

```
If (New_Start < Existing_End) AND (Existing_Start < New_End)
  → Conflict exists and booking is rejected
```

## 🧪 Testing Checklist

- [ ] Register a new user
- [ ] Login with valid credentials
- [ ] Book an amenity and verify it appears under bookings
- [ ] Attempt to book overlapping time (should fail)
- [ ] Add a community resource
- [ ] Request to borrow another resident's resource
- [ ] Submit a new complaint
- [ ] Check if admin can update complaint status
- [ ] View announcements posted by admin
- [ ] Confirm notifications appear as expected

## 🔐 Environment Variables

```env
MONGO_URI=mongodb://mongo:27017/smart_community
PORT=4000
JWT_SECRET=your_jwt_secret_key_here
```

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

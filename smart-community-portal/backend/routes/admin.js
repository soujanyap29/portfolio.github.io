const express = require('express');
const router = express.Router();
const User = require('../models/User');
const Booking = require('../models/Booking');
const Complaint = require('../models/Complaint');
const Resource = require('../models/Resource');
const { protect, admin } = require('../middleware/auth');

router.get('/users', protect, admin, async (req, res) => {
    try {
        const users = await User.find().select('-password').sort({ createdAt: -1 });
        res.json(users);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/users/:id', protect, admin, async (req, res) => {
    try {
        const user = await User.findById(req.params.id).select('-password');
        
        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        const bookings = await Booking.find({ user: req.params.id })
            .populate('amenity', 'name')
            .sort({ date: -1 })
            .limit(10);

        const complaints = await Complaint.find({ user: req.params.id })
            .sort({ createdAt: -1 })
            .limit(10);

        const resources = await Resource.find({ owner: req.params.id })
            .sort({ createdAt: -1 });

        res.json({
            user,
            bookings,
            complaints,
            resources
        });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/users/:id/role', protect, admin, async (req, res) => {
    try {
        const user = await User.findById(req.params.id);

        if (!user) {
            return res.status(404).json({ message: 'User not found' });
        }

        if (user._id.toString() === req.user._id.toString()) {
            return res.status(400).json({ message: 'Cannot change your own role' });
        }

        user.role = req.body.role;
        await user.save();

        res.json({ message: 'User role updated successfully', user: { _id: user._id, role: user.role } });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/stats', protect, admin, async (req, res) => {
    try {
        const totalUsers = await User.countDocuments();
        const totalResidents = await User.countDocuments({ role: 'resident' });
        const totalAdmins = await User.countDocuments({ role: 'admin' });

        const totalBookings = await Booking.countDocuments();
        const confirmedBookings = await Booking.countDocuments({ status: 'confirmed' });

        const totalComplaints = await Complaint.countDocuments();
        const submittedComplaints = await Complaint.countDocuments({ status: 'submitted' });
        const inProgressComplaints = await Complaint.countDocuments({ status: 'in_progress' });
        const resolvedComplaints = await Complaint.countDocuments({ status: 'resolved' });

        const totalResources = await Resource.countDocuments();
        const availableResources = await Resource.countDocuments({ status: 'available' });
        const borrowedResources = await Resource.countDocuments({ status: 'borrowed' });

        res.json({
            users: {
                total: totalUsers,
                residents: totalResidents,
                admins: totalAdmins
            },
            bookings: {
                total: totalBookings,
                confirmed: confirmedBookings
            },
            complaints: {
                total: totalComplaints,
                submitted: submittedComplaints,
                inProgress: inProgressComplaints,
                resolved: resolvedComplaints
            },
            resources: {
                total: totalResources,
                available: availableResources,
                borrowed: borrowedResources
            }
        });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

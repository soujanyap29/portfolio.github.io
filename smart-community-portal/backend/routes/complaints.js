const express = require('express');
const router = express.Router();
const Complaint = require('../models/Complaint');
const Notification = require('../models/Notification');
const { protect, admin } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        let query = {};

        if (req.user.role !== 'admin') {
            query.user = req.user._id;
        }

        if (req.query.status) {
            query.status = req.query.status;
        }

        if (req.query.category) {
            query.category = req.query.category;
        }

        const complaints = await Complaint.find(query)
            .populate('user', 'name flatNumber email')
            .sort({ createdAt: -1 });

        res.json(complaints);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/my-complaints', protect, async (req, res) => {
    try {
        const complaints = await Complaint.find({ user: req.user._id })
            .sort({ createdAt: -1 });

        res.json(complaints);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/:id', protect, async (req, res) => {
    try {
        const complaint = await Complaint.findById(req.params.id)
            .populate('user', 'name flatNumber email phone');

        if (!complaint) {
            return res.status(404).json({ message: 'Complaint not found' });
        }

        if (complaint.user._id.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to view this complaint' });
        }

        res.json(complaint);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, async (req, res) => {
    try {
        const { title, description, category, imageUrl, priority } = req.body;

        const complaint = await Complaint.create({
            user: req.user._id,
            title,
            description,
            category,
            imageUrl,
            priority
        });

        await Notification.create({
            user: req.user._id,
            title: 'Complaint Submitted',
            message: `Your complaint "${title}" has been submitted and is being reviewed.`,
            type: 'complaint',
            relatedId: complaint._id,
            relatedModel: 'Complaint'
        });

        const populatedComplaint = await Complaint.findById(complaint._id)
            .populate('user', 'name flatNumber');

        res.status(201).json(populatedComplaint);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id', protect, async (req, res) => {
    try {
        const complaint = await Complaint.findById(req.params.id);

        if (!complaint) {
            return res.status(404).json({ message: 'Complaint not found' });
        }

        if (complaint.user.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to update this complaint' });
        }

        if (req.user.role !== 'admin' && complaint.status !== 'submitted') {
            return res.status(400).json({ message: 'Can only edit complaints that are still in submitted status' });
        }

        const { title, description, category, imageUrl, priority } = req.body;

        complaint.title = title || complaint.title;
        complaint.description = description || complaint.description;
        complaint.category = category || complaint.category;
        complaint.imageUrl = imageUrl !== undefined ? imageUrl : complaint.imageUrl;
        complaint.priority = priority || complaint.priority;

        const updatedComplaint = await complaint.save();

        res.json(updatedComplaint);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id/status', protect, admin, async (req, res) => {
    try {
        const complaint = await Complaint.findById(req.params.id);

        if (!complaint) {
            return res.status(404).json({ message: 'Complaint not found' });
        }

        const { status, adminNotes } = req.body;

        const validTransitions = {
            'submitted': ['in_progress', 'resolved'],
            'in_progress': ['resolved'],
            'resolved': []
        };

        if (!validTransitions[complaint.status].includes(status)) {
            return res.status(400).json({ 
                message: `Invalid status transition from ${complaint.status} to ${status}` 
            });
        }

        complaint.status = status;
        complaint.adminNotes = adminNotes || complaint.adminNotes;

        if (status === 'resolved') {
            complaint.resolvedAt = new Date();
        }

        const updatedComplaint = await complaint.save();

        const statusMessages = {
            'in_progress': 'Your complaint is now being processed.',
            'resolved': 'Your complaint has been resolved.'
        };

        await Notification.create({
            user: complaint.user,
            title: 'Complaint Status Updated',
            message: `Complaint "${complaint.title}": ${statusMessages[status]}`,
            type: 'complaint',
            relatedId: complaint._id,
            relatedModel: 'Complaint'
        });

        const populatedComplaint = await Complaint.findById(updatedComplaint._id)
            .populate('user', 'name flatNumber email');

        res.json(populatedComplaint);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, async (req, res) => {
    try {
        const complaint = await Complaint.findById(req.params.id);

        if (!complaint) {
            return res.status(404).json({ message: 'Complaint not found' });
        }

        if (complaint.user.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to delete this complaint' });
        }

        if (complaint.status !== 'submitted') {
            return res.status(400).json({ message: 'Can only delete complaints that are still in submitted status' });
        }

        await Complaint.findByIdAndDelete(req.params.id);
        res.json({ message: 'Complaint deleted successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

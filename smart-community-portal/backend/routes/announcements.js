const express = require('express');
const router = express.Router();
const Announcement = require('../models/Announcement');
const Notification = require('../models/Notification');
const User = require('../models/User');
const { protect, admin } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        let query = { isActive: true };

        if (req.query.category) {
            query.category = req.query.category;
        }

        const now = new Date();
        query.$or = [
            { expiresAt: null },
            { expiresAt: { $gt: now } }
        ];

        const announcements = await Announcement.find(query)
            .populate('author', 'name')
            .sort({ priority: -1, createdAt: -1 });

        res.json(announcements);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/all', protect, admin, async (req, res) => {
    try {
        const announcements = await Announcement.find()
            .populate('author', 'name')
            .sort({ createdAt: -1 });

        res.json(announcements);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/:id', protect, async (req, res) => {
    try {
        const announcement = await Announcement.findById(req.params.id)
            .populate('author', 'name');

        if (announcement) {
            res.json(announcement);
        } else {
            res.status(404).json({ message: 'Announcement not found' });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, admin, async (req, res) => {
    try {
        const { title, content, category, priority, expiresAt } = req.body;

        const announcement = await Announcement.create({
            title,
            content,
            category,
            priority,
            author: req.user._id,
            expiresAt
        });

        const users = await User.find({ _id: { $ne: req.user._id } });
        
        const notifications = users.map(user => ({
            user: user._id,
            title: 'New Announcement',
            message: `${title}`,
            type: 'announcement',
            relatedId: announcement._id,
            relatedModel: 'Announcement'
        }));

        await Notification.insertMany(notifications);

        const populatedAnnouncement = await Announcement.findById(announcement._id)
            .populate('author', 'name');

        res.status(201).json(populatedAnnouncement);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id', protect, admin, async (req, res) => {
    try {
        const announcement = await Announcement.findById(req.params.id);

        if (!announcement) {
            return res.status(404).json({ message: 'Announcement not found' });
        }

        const { title, content, category, priority, expiresAt, isActive } = req.body;

        announcement.title = title || announcement.title;
        announcement.content = content || announcement.content;
        announcement.category = category || announcement.category;
        announcement.priority = priority || announcement.priority;
        announcement.expiresAt = expiresAt !== undefined ? expiresAt : announcement.expiresAt;
        announcement.isActive = isActive !== undefined ? isActive : announcement.isActive;

        const updatedAnnouncement = await announcement.save();

        const populatedAnnouncement = await Announcement.findById(updatedAnnouncement._id)
            .populate('author', 'name');

        res.json(populatedAnnouncement);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, admin, async (req, res) => {
    try {
        const announcement = await Announcement.findById(req.params.id);

        if (!announcement) {
            return res.status(404).json({ message: 'Announcement not found' });
        }

        await Announcement.findByIdAndDelete(req.params.id);
        res.json({ message: 'Announcement deleted successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

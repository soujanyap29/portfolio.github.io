const express = require('express');
const router = express.Router();
const Notification = require('../models/Notification');
const { protect } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        let query = { user: req.user._id };

        if (req.query.unread === 'true') {
            query.isRead = false;
        }

        const notifications = await Notification.find(query)
            .sort({ createdAt: -1 })
            .limit(50);

        res.json(notifications);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/unread-count', protect, async (req, res) => {
    try {
        const count = await Notification.countDocuments({
            user: req.user._id,
            isRead: false
        });

        res.json({ count });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id/read', protect, async (req, res) => {
    try {
        const notification = await Notification.findById(req.params.id);

        if (!notification) {
            return res.status(404).json({ message: 'Notification not found' });
        }

        if (notification.user.toString() !== req.user._id.toString()) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        notification.isRead = true;
        await notification.save();

        res.json(notification);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/read-all', protect, async (req, res) => {
    try {
        await Notification.updateMany(
            { user: req.user._id, isRead: false },
            { isRead: true }
        );

        res.json({ message: 'All notifications marked as read' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, async (req, res) => {
    try {
        const notification = await Notification.findById(req.params.id);

        if (!notification) {
            return res.status(404).json({ message: 'Notification not found' });
        }

        if (notification.user.toString() !== req.user._id.toString()) {
            return res.status(403).json({ message: 'Not authorized' });
        }

        await Notification.findByIdAndDelete(req.params.id);
        res.json({ message: 'Notification deleted' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/', protect, async (req, res) => {
    try {
        await Notification.deleteMany({ user: req.user._id });
        res.json({ message: 'All notifications deleted' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

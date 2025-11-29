const express = require('express');
const router = express.Router();
const BorrowRequest = require('../models/BorrowRequest');
const Resource = require('../models/Resource');
const Notification = require('../models/Notification');
const { protect } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        let query = {};

        if (req.query.type === 'received') {
            query.owner = req.user._id;
        } else if (req.query.type === 'sent') {
            query.borrower = req.user._id;
        } else {
            query.$or = [
                { owner: req.user._id },
                { borrower: req.user._id }
            ];
        }

        if (req.query.status) {
            query.status = req.query.status;
        }

        const requests = await BorrowRequest.find(query)
            .populate('resource', 'name description category imageUrl')
            .populate('borrower', 'name flatNumber email phone')
            .populate('owner', 'name flatNumber')
            .sort({ createdAt: -1 });

        res.json(requests);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, async (req, res) => {
    try {
        const { resourceId, message, borrowDate, returnDate } = req.body;

        const resource = await Resource.findById(resourceId);
        if (!resource) {
            return res.status(404).json({ message: 'Resource not found' });
        }

        if (resource.status !== 'available') {
            return res.status(400).json({ message: 'Resource is not available for borrowing' });
        }

        if (resource.owner.toString() === req.user._id.toString()) {
            return res.status(400).json({ message: 'You cannot borrow your own resource' });
        }

        const existingRequest = await BorrowRequest.findOne({
            resource: resourceId,
            borrower: req.user._id,
            status: 'pending'
        });

        if (existingRequest) {
            return res.status(400).json({ message: 'You already have a pending request for this resource' });
        }

        const borrowRequest = await BorrowRequest.create({
            resource: resourceId,
            borrower: req.user._id,
            owner: resource.owner,
            message,
            borrowDate,
            returnDate
        });

        await Notification.create({
            user: resource.owner,
            title: 'New Borrow Request',
            message: `${req.user.name} has requested to borrow your ${resource.name}.`,
            type: 'resource',
            relatedId: borrowRequest._id,
            relatedModel: 'BorrowRequest'
        });

        const populatedRequest = await BorrowRequest.findById(borrowRequest._id)
            .populate('resource', 'name description')
            .populate('borrower', 'name flatNumber')
            .populate('owner', 'name flatNumber');

        res.status(201).json(populatedRequest);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id/approve', protect, async (req, res) => {
    try {
        const borrowRequest = await BorrowRequest.findById(req.params.id)
            .populate('resource');

        if (!borrowRequest) {
            return res.status(404).json({ message: 'Borrow request not found' });
        }

        if (borrowRequest.owner.toString() !== req.user._id.toString()) {
            return res.status(403).json({ message: 'Not authorized to approve this request' });
        }

        if (borrowRequest.status !== 'pending') {
            return res.status(400).json({ message: 'This request has already been processed' });
        }

        borrowRequest.status = 'approved';
        await borrowRequest.save();

        const resource = await Resource.findById(borrowRequest.resource._id);
        resource.status = 'borrowed';
        resource.currentBorrower = borrowRequest.borrower;
        await resource.save();

        await BorrowRequest.updateMany(
            {
                resource: borrowRequest.resource._id,
                _id: { $ne: borrowRequest._id },
                status: 'pending'
            },
            { status: 'rejected' }
        );

        await Notification.create({
            user: borrowRequest.borrower,
            title: 'Borrow Request Approved',
            message: `Your request to borrow ${borrowRequest.resource.name} has been approved.`,
            type: 'resource',
            relatedId: borrowRequest._id,
            relatedModel: 'BorrowRequest'
        });

        const populatedRequest = await BorrowRequest.findById(borrowRequest._id)
            .populate('resource', 'name description')
            .populate('borrower', 'name flatNumber email phone')
            .populate('owner', 'name flatNumber');

        res.json(populatedRequest);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id/reject', protect, async (req, res) => {
    try {
        const borrowRequest = await BorrowRequest.findById(req.params.id)
            .populate('resource');

        if (!borrowRequest) {
            return res.status(404).json({ message: 'Borrow request not found' });
        }

        if (borrowRequest.owner.toString() !== req.user._id.toString()) {
            return res.status(403).json({ message: 'Not authorized to reject this request' });
        }

        if (borrowRequest.status !== 'pending') {
            return res.status(400).json({ message: 'This request has already been processed' });
        }

        borrowRequest.status = 'rejected';
        await borrowRequest.save();

        await Notification.create({
            user: borrowRequest.borrower,
            title: 'Borrow Request Rejected',
            message: `Your request to borrow ${borrowRequest.resource.name} has been rejected.`,
            type: 'resource',
            relatedId: borrowRequest._id,
            relatedModel: 'BorrowRequest'
        });

        res.json({ message: 'Request rejected successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id/return', protect, async (req, res) => {
    try {
        const borrowRequest = await BorrowRequest.findById(req.params.id)
            .populate('resource');

        if (!borrowRequest) {
            return res.status(404).json({ message: 'Borrow request not found' });
        }

        if (borrowRequest.borrower.toString() !== req.user._id.toString() && 
            borrowRequest.owner.toString() !== req.user._id.toString()) {
            return res.status(403).json({ message: 'Not authorized to mark this as returned' });
        }

        if (borrowRequest.status !== 'approved') {
            return res.status(400).json({ message: 'This resource is not currently borrowed' });
        }

        borrowRequest.status = 'returned';
        borrowRequest.actualReturnDate = new Date();
        await borrowRequest.save();

        const resource = await Resource.findById(borrowRequest.resource._id);
        resource.status = 'available';
        resource.currentBorrower = null;
        await resource.save();

        await Notification.create({
            user: borrowRequest.owner,
            title: 'Resource Returned',
            message: `${borrowRequest.resource.name} has been marked as returned.`,
            type: 'resource',
            relatedId: borrowRequest._id,
            relatedModel: 'BorrowRequest'
        });

        res.json({ message: 'Resource marked as returned successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

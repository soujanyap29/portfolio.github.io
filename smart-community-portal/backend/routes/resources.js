const express = require('express');
const router = express.Router();
const Resource = require('../models/Resource');
const { protect, admin } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        let query = {};

        if (req.query.status) {
            query.status = req.query.status;
        }

        if (req.query.category) {
            query.category = req.query.category;
        }

        const resources = await Resource.find(query)
            .populate('owner', 'name flatNumber')
            .populate('currentBorrower', 'name flatNumber')
            .sort({ createdAt: -1 });

        res.json(resources);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/my-resources', protect, async (req, res) => {
    try {
        const resources = await Resource.find({ owner: req.user._id })
            .populate('currentBorrower', 'name flatNumber')
            .sort({ createdAt: -1 });

        res.json(resources);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/:id', protect, async (req, res) => {
    try {
        const resource = await Resource.findById(req.params.id)
            .populate('owner', 'name flatNumber email phone')
            .populate('currentBorrower', 'name flatNumber');

        if (resource) {
            res.json(resource);
        } else {
            res.status(404).json({ message: 'Resource not found' });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, async (req, res) => {
    try {
        const { name, description, category, imageUrl, availableFrom, availableTo } = req.body;

        const resource = await Resource.create({
            owner: req.user._id,
            name,
            description,
            category,
            imageUrl,
            availableFrom,
            availableTo
        });

        const populatedResource = await Resource.findById(resource._id)
            .populate('owner', 'name flatNumber');

        res.status(201).json(populatedResource);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id', protect, async (req, res) => {
    try {
        const resource = await Resource.findById(req.params.id);

        if (!resource) {
            return res.status(404).json({ message: 'Resource not found' });
        }

        if (resource.owner.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to update this resource' });
        }

        const { name, description, category, imageUrl, availableFrom, availableTo, status } = req.body;

        resource.name = name || resource.name;
        resource.description = description || resource.description;
        resource.category = category || resource.category;
        resource.imageUrl = imageUrl !== undefined ? imageUrl : resource.imageUrl;
        resource.availableFrom = availableFrom || resource.availableFrom;
        resource.availableTo = availableTo || resource.availableTo;
        resource.status = status || resource.status;

        const updatedResource = await resource.save();

        const populatedResource = await Resource.findById(updatedResource._id)
            .populate('owner', 'name flatNumber')
            .populate('currentBorrower', 'name flatNumber');

        res.json(populatedResource);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, async (req, res) => {
    try {
        const resource = await Resource.findById(req.params.id);

        if (!resource) {
            return res.status(404).json({ message: 'Resource not found' });
        }

        if (resource.owner.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to delete this resource' });
        }

        if (resource.status === 'borrowed') {
            return res.status(400).json({ message: 'Cannot delete a resource that is currently borrowed' });
        }

        await Resource.findByIdAndDelete(req.params.id);
        res.json({ message: 'Resource deleted successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

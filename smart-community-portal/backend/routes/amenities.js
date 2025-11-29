const express = require('express');
const router = express.Router();
const Amenity = require('../models/Amenity');
const { protect, admin } = require('../middleware/auth');

router.get('/', protect, async (req, res) => {
    try {
        const amenities = await Amenity.find({ isActive: true });
        res.json(amenities);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/:id', protect, async (req, res) => {
    try {
        const amenity = await Amenity.findById(req.params.id);
        if (amenity) {
            res.json(amenity);
        } else {
            res.status(404).json({ message: 'Amenity not found' });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, admin, async (req, res) => {
    try {
        const { name, description, location, operatingHours, capacity } = req.body;

        const amenity = await Amenity.create({
            name,
            description,
            location,
            operatingHours,
            capacity
        });

        res.status(201).json(amenity);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id', protect, admin, async (req, res) => {
    try {
        const amenity = await Amenity.findById(req.params.id);

        if (amenity) {
            amenity.name = req.body.name || amenity.name;
            amenity.description = req.body.description || amenity.description;
            amenity.location = req.body.location || amenity.location;
            amenity.operatingHours = req.body.operatingHours || amenity.operatingHours;
            amenity.capacity = req.body.capacity || amenity.capacity;
            amenity.isActive = req.body.isActive !== undefined ? req.body.isActive : amenity.isActive;

            const updatedAmenity = await amenity.save();
            res.json(updatedAmenity);
        } else {
            res.status(404).json({ message: 'Amenity not found' });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, admin, async (req, res) => {
    try {
        const amenity = await Amenity.findById(req.params.id);

        if (amenity) {
            amenity.isActive = false;
            await amenity.save();
            res.json({ message: 'Amenity deactivated' });
        } else {
            res.status(404).json({ message: 'Amenity not found' });
        }
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

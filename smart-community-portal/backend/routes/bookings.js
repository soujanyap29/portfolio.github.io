const express = require('express');
const router = express.Router();
const Booking = require('../models/Booking');
const Amenity = require('../models/Amenity');
const Notification = require('../models/Notification');
const { protect, admin } = require('../middleware/auth');

const checkBookingConflict = async (amenityId, date, startTime, endTime, excludeBookingId = null) => {
    const dateOnly = new Date(date);
    dateOnly.setHours(0, 0, 0, 0);
    
    const nextDay = new Date(dateOnly);
    nextDay.setDate(nextDay.getDate() + 1);

    const query = {
        amenity: amenityId,
        date: { $gte: dateOnly, $lt: nextDay },
        status: 'confirmed'
    };

    if (excludeBookingId) {
        query._id = { $ne: excludeBookingId };
    }

    const existingBookings = await Booking.find(query);

    const newStart = parseInt(startTime.replace(':', ''));
    const newEnd = parseInt(endTime.replace(':', ''));

    for (const booking of existingBookings) {
        const existingStart = parseInt(booking.startTime.replace(':', ''));
        const existingEnd = parseInt(booking.endTime.replace(':', ''));

        if (newStart < existingEnd && existingStart < newEnd) {
            return true;
        }
    }

    return false;
};

router.get('/', protect, async (req, res) => {
    try {
        let query = {};
        
        if (req.user.role !== 'admin') {
            query.user = req.user._id;
        }

        if (req.query.amenity) {
            query.amenity = req.query.amenity;
        }

        if (req.query.status) {
            query.status = req.query.status;
        }

        const bookings = await Booking.find(query)
            .populate('user', 'name email flatNumber')
            .populate('amenity', 'name location')
            .sort({ date: -1 });

        res.json(bookings);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/my-bookings', protect, async (req, res) => {
    try {
        const bookings = await Booking.find({ user: req.user._id })
            .populate('amenity', 'name location')
            .sort({ date: -1 });

        res.json(bookings);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/amenity/:amenityId', protect, async (req, res) => {
    try {
        const { date } = req.query;
        let query = { amenity: req.params.amenityId, status: 'confirmed' };

        if (date) {
            const dateOnly = new Date(date);
            dateOnly.setHours(0, 0, 0, 0);
            
            const nextDay = new Date(dateOnly);
            nextDay.setDate(nextDay.getDate() + 1);

            query.date = { $gte: dateOnly, $lt: nextDay };
        }

        const bookings = await Booking.find(query)
            .populate('user', 'name flatNumber')
            .sort({ startTime: 1 });

        res.json(bookings);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.post('/', protect, async (req, res) => {
    try {
        const { amenityId, date, startTime, endTime, notes } = req.body;

        const amenity = await Amenity.findById(amenityId);
        if (!amenity || !amenity.isActive) {
            return res.status(404).json({ message: 'Amenity not found or inactive' });
        }

        const hasConflict = await checkBookingConflict(amenityId, date, startTime, endTime);
        if (hasConflict) {
            return res.status(400).json({ 
                message: 'Booking conflict: The selected time slot overlaps with an existing booking' 
            });
        }

        const booking = await Booking.create({
            user: req.user._id,
            amenity: amenityId,
            date: new Date(date),
            startTime,
            endTime,
            notes
        });

        await Notification.create({
            user: req.user._id,
            title: 'Booking Confirmed',
            message: `Your booking for ${amenity.name} on ${new Date(date).toLocaleDateString()} from ${startTime} to ${endTime} has been confirmed.`,
            type: 'booking',
            relatedId: booking._id,
            relatedModel: 'Booking'
        });

        const populatedBooking = await Booking.findById(booking._id)
            .populate('user', 'name email flatNumber')
            .populate('amenity', 'name location');

        res.status(201).json(populatedBooking);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.put('/:id', protect, async (req, res) => {
    try {
        const booking = await Booking.findById(req.params.id);

        if (!booking) {
            return res.status(404).json({ message: 'Booking not found' });
        }

        if (booking.user.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to update this booking' });
        }

        const { date, startTime, endTime, notes, status } = req.body;

        if (date || startTime || endTime) {
            const hasConflict = await checkBookingConflict(
                booking.amenity,
                date || booking.date,
                startTime || booking.startTime,
                endTime || booking.endTime,
                booking._id
            );

            if (hasConflict) {
                return res.status(400).json({ 
                    message: 'Booking conflict: The selected time slot overlaps with an existing booking' 
                });
            }
        }

        booking.date = date ? new Date(date) : booking.date;
        booking.startTime = startTime || booking.startTime;
        booking.endTime = endTime || booking.endTime;
        booking.notes = notes !== undefined ? notes : booking.notes;
        booking.status = status || booking.status;

        const updatedBooking = await booking.save();
        
        const populatedBooking = await Booking.findById(updatedBooking._id)
            .populate('user', 'name email flatNumber')
            .populate('amenity', 'name location');

        res.json(populatedBooking);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.delete('/:id', protect, async (req, res) => {
    try {
        const booking = await Booking.findById(req.params.id).populate('amenity', 'name');

        if (!booking) {
            return res.status(404).json({ message: 'Booking not found' });
        }

        if (booking.user.toString() !== req.user._id.toString() && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to cancel this booking' });
        }

        booking.status = 'cancelled';
        await booking.save();

        await Notification.create({
            user: booking.user,
            title: 'Booking Cancelled',
            message: `Your booking for ${booking.amenity.name} on ${booking.date.toLocaleDateString()} has been cancelled.`,
            type: 'booking',
            relatedId: booking._id,
            relatedModel: 'Booking'
        });

        res.json({ message: 'Booking cancelled successfully' });
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

module.exports = router;

import React from 'react';

const FormInputs = ({ formData, handleChange }) => {
  return (
    <div>
      <div className="form-group">
        <label htmlFor="base_price">Base Price</label>
        <input
          id="base_price"
          name="base_price"
          type="number"
          className="input-field"
          value={formData.base_price}
          onChange={handleChange}
          placeholder="Enter base price..."
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="total_price">Total Price</label>
        <input
          id="total_price"
          name="total_price"
          type="number"
          className="input-field"
          value={formData.total_price}
          onChange={handleChange}
          placeholder="Enter total price..."
          required
        />
      </div>

      <div className="form-group">
        <label htmlFor="is_featured_sku">Is Featured SKU</label>
        <select
          id="is_featured_sku"
          name="is_featured_sku"
          className="input-field"
          value={formData.is_featured_sku ? "yes" : "no"}
          onChange={(e) =>
            handleChange({
              target: {
                name: e.target.name,
                value: e.target.value === "yes",
              },
            })
          }
          required
        >
          <option value="">Select...</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="is_display_sku">Is Display SKU</label>
        <select
          id="is_display_sku"
          name="is_display_sku"
          className="input-field"
          value={formData.is_display_sku ? "yes" : "no"}
          onChange={(e) =>
            handleChange({
              target: {
                name: e.target.name,
                value: e.target.value === "yes",
              },
            })
          }
          required
        >
          <option value="">Select...</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
      </div>

      <div className="form-group">
        <label htmlFor="sku_id">SKU ID</label>
        <input
          id="sku_id"
          name="sku_id"
          type="text"
          className="input-field"
          value={formData.sku_id}
          onChange={handleChange}
          placeholder="Enter SKU ID..."
          required
        />
      </div>
    </div>
  );
};

export default FormInputs;
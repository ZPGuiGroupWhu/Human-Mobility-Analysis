import { createAsyncThunk } from "@reduxjs/toolkit";

// 异步请求
export const fetchPOI = createAsyncThunk("common/fetchPOI", async () => {
    const { data } = await axios.get(
      `${process.env.PUBLIC_URL}/mock/shenzhen-poi.json`
    );
    const result = data.map((item) => {
      const { typeId, location, ...subItem } = item;
      return {
        ...subItem,
        count: 1,
        typeId: +typeId,
        location: location.split(",").map((item) => +parseFloat(item).toFixed(4)),
      };
    });
    console.log('fetch poi is', result);
    return result;
  });
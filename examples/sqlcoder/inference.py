from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.20.44:7861/v1",
    api_key = "EMPTY"
)


SQL_PROMPT = """### Instructions:
Your task is to convert a question into a SQL query, given a database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
- When creating a ratio, always cast the numerator as float

### Input:
Generate a SQL query that answers the question `{question}`.
This query will run on a database whose schema is represented in this string:
{database_schema}

### Response:
Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
```sql
"""

database_schema1 = """CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);

CREATE TABLE customers (
   customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
   name VARCHAR(50), -- Name of the customer
   address VARCHAR(100) -- Mailing address of the customer
);

CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred
  quantity INTEGER -- Quantity of product sold
);

CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);

-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id
"""

question1 = "What product has the biggest fall in sales in 2022 compared to 2021? Give me the product name, the sales amount in both years, and the difference."

database_schema2 = """CREATE TABLE `institution` (
  `institution_id` int NOT NULL AUTO_INCREMENT,
  `institution_number` bigint DEFAULT NULL COMMENT '机构编号唯一',
  `institution_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '机构名称(包括：学校，民营组织等)',
  `institution_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL COMMENT '机构类型（如高等院校、事业单位等）',
  `institution_province` varchar(255) DEFAULT NULL COMMENT '省份',
  `institution_city` varchar(255) DEFAULT NULL COMMENT '城市',
  `institution_district` varchar(255) DEFAULT NULL COMMENT '行政区',
);
"""

question2 = "哪个城市的高校数目最多？"


def test():
    for q, s in zip([question1, question2], [database_schema1, database_schema2]):
        prompt = SQL_PROMPT.format(question=q, database_schema=s)
        print(f"Qusestion: \n{q}")
        response = client.completions.create(
            model="sqlcoder",
            prompt=prompt,
            temperature=0,
            stop=["```"],
        )
        output = response.choices[0].text.split("```sql")[-1].split("```")[0].split(";")[0].strip() + ";"
        print(f"SQL: \n{output}\n\n")


def run_codeqwen():
    for q, s in zip([question1, question2], [database_schema1, database_schema2]):
        prompt = SQL_PROMPT.format(question=q, database_schema=s)
        print(f"Qusestion: \n{q}")
        response = client.chat.completions.create(
            model="codeqwen",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=["```"],
        )
        output = response.choices[0].message.content
        print(f"SQL: \n{output}\n\n")


if __name__ == "__main__":
    run_codeqwen()

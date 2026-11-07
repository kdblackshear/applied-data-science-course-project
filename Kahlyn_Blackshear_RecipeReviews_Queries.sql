-- 1. Record counts across all relational tables
SELECT 'recipes' AS table_name, COUNT(*) AS record_count FROM recipes
UNION ALL
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'reviews', COUNT(*) FROM reviews;


-- 2. Filtering records using WHERE (Isolating top 5-star reviews)
SELECT comment_id, stars, text 
FROM reviews 
WHERE stars = 5 
LIMIT 5;


-- 3. Comparing average ratings across recipes using GROUP BY and AVG()
SELECT r.recipe_name, 
       ROUND(AVG(rev.stars), 2) AS avg_rating, 
       COUNT(rev.comment_id) AS total_reviews
FROM recipes r
INNER JOIN reviews rev ON r.recipe_code = rev.recipe_code
GROUP BY r.recipe_name;


-- 4. Identifying users with a specified minimum number of reviews using GROUP BY, COUNT(), and HAVING
SELECT user_id, COUNT(*) AS review_count
FROM reviews
GROUP BY user_id
HAVING COUNT(*) >= 2
ORDER BY review_count DESC;


-- 5. INNER JOIN between recipes and reviews tables
SELECT rev.comment_id, r.recipe_name, rev.stars, rev.thumbs_up
FROM reviews rev
INNER JOIN recipes r ON rev.recipe_code = r.recipe_code
LIMIT 5;


-- 6. Joining all three relational tables (recipes, users, and reviews)
SELECT rev.comment_id, u.user_name, r.recipe_name, rev.stars, rev.created_at
FROM reviews rev
INNER JOIN users u ON rev.user_id = u.user_id
INNER JOIN recipes r ON rev.recipe_code = r.recipe_code
ORDER BY rev.created_at DESC
LIMIT 5;


-- 7. Original Research Query: Impact of User Reputation Tiers on Review Ratings
SELECT 
    CASE 
        WHEN user_reputation = 1 THEN 'New/Base User (Rep 1)'
        WHEN user_reputation BETWEEN 2 AND 20 THEN 'Moderate Reputation (2-20)'
        ELSE 'High Reputation (21+)'
    END AS reputation_tier,
    COUNT(*) AS total_reviews,
    ROUND(AVG(stars), 2) AS avg_rating,
    ROUND(AVG(thumbs_up), 2) AS avg_thumbs_up
FROM reviews
GROUP BY reputation_tier
ORDER BY avg_rating DESC;